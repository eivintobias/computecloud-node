"""Security monitor for session containers (Tier 3 security).

Detects crypto-mining and other anomalous activity inside Docker
containers by scanning running processes and CPU usage patterns.

Used by the node client during the session heartbeat loop: every
``security_scan_interval_seconds`` the monitor runs
``docker top {container_id}`` and checks the process list against a
blocklist of known miner names.  It also samples CPU usage via
``docker stats`` and flags sustained 100% usage as suspicious.

If a miner or anomaly is detected, the caller terminates the session
and reports the security violation to the server.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Known crypto-miner process names (lower-cased for matching).
KNOWN_MINER_PROCESSES: list[str] = [
    "xmrig",
    "ethminer",
    "cgminer",
    "bfgminer",
    "minerd",
    "cpuminer",
    "ccminer",
    "t-rex",
    "trex",
    "teamredminer",
    "nbminer",
    "phoenixminer",
    "lolminer",
    "gminer",
    "tminer",
    "minergate",
    "nicehashminer",
    "claymore",
    "ethashminer",
    "progpowminer",
    "cryptonight",
    "stratum",
    "nsfminer",
    "wildrig",
    "xmr-stak",
    "xmrig-amd",
    "xmrig-cpu",
    "xmrig-nvidia",
    "cryptodredge",
    "mkxminer",
    "sgminer",
    "zm",
    "z-enemy",
]

# Known mining pool domains to block via Docker --add-host.
# These are redirected to 127.0.0.1 to prevent the container from
# connecting to mining pools.
KNOWN_MINING_POOL_DOMAINS: list[str] = [
    "pool.minexmr.com",
    "pool.supportxmr.com",
    "xmr.pool.minergate.com",
    "eth.pool.minergate.com",
    "btc.pool.minergate.com",
    "pool.moneropool.com",
    "xmrpool.eu",
    "monero.hashpool.pro",
    "nanopool.org",
    "ethermine.org",
    "f2pool.com",
    "antpool.com",
    "viabtc.com",
    "miningrigrentals.com",
    "nicehash.com",
    "miningpoolhub.com",
    "dwarfpool.com",
    "flypool.org",
    "nanopool.com",
    "2miners.com",
    "mine.xmr.ru",
    "monero.crypto-pool.org",
]


@dataclass
class SecurityAlert:
    """Result of a security scan."""

    alert_type: str  # "mining_detected", "anomaly_cpu", "suspicious_process"
    severity: str  # "critical", "warning", "info"
    details: str = ""
    process_names: list[str] = field(default_factory=list)
    cpu_percent: float = 0.0


class SecurityMonitor:
    """Periodic security scanner for session containers.

    Maintains a rolling CPU usage history to detect sustained high
    usage (an anomaly that may indicate crypto mining).
    """

    def __init__(
        self,
        cpu_threshold_percent: float = 95.0,
        sustained_checks: int = 10,
    ) -> None:
        self.cpu_threshold = cpu_threshold_percent
        self.sustained_checks = sustained_checks
        self._cpu_history: list[float] = []  # rolling history of CPU %

    def scan_container(self, container_id: str) -> SecurityAlert | None:
        """Run a full security scan on a container.

        Returns a :class:`SecurityAlert` if a violation is detected,
        or ``None`` if the container looks clean.
        """
        # 1. Check for known miner processes.
        proc_names = self._get_container_processes(container_id)
        miners_found = self._check_for_miners(proc_names)
        if miners_found:
            return SecurityAlert(
                alert_type="mining_detected",
                severity="critical",
                details=f"Known miner process(es) detected: {miners_found}",
                process_names=miners_found,
            )

        # 2. Check CPU usage pattern.
        cpu_percent = self._get_container_cpu(container_id)
        self._cpu_history.append(cpu_percent)
        # Keep only the last N samples.
        if len(self._cpu_history) > self.sustained_checks:
            self._cpu_history = self._cpu_history[-self.sustained_checks:]

        # If we have enough samples and ALL exceed the threshold, flag it.
        if (
            len(self._cpu_history) >= self.sustained_checks
            and all(c >= self.cpu_threshold for c in self._cpu_history)
        ):
            return SecurityAlert(
                alert_type="anomaly_cpu",
                severity="critical",
                details=(
                    f"Sustained CPU >= {self.cpu_threshold}% for "
                    f"{len(self._cpu_history)} consecutive checks"
                ),
                cpu_percent=cpu_percent,
            )

        # 3. Check for suspicious process names.
        suspicious = self._check_suspicious_processes(proc_names)
        if suspicious:
            return SecurityAlert(
                alert_type="suspicious_process",
                severity="warning",
                details=f"Suspicious process(es): {suspicious}",
                process_names=suspicious,
            )

    @staticmethod
    def _get_container_processes(container_id: str) -> list[str]:
        """Get the list of process names running inside a container."""
        try:
            result = subprocess.run(
                ["docker", "top", container_id, "-o", "pid,comm"],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode != 0:
                return []
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return []
            procs = []
            for line in lines[1:]:
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    procs.append(parts[1].strip())
            return procs
        except Exception:
            return []

    @staticmethod
    def _check_for_miners(process_names: list[str]) -> list[str]:
        """Check process names against the known miner blocklist."""
        found = []
        for proc in process_names:
            proc_lower = proc.lower()
            for miner in KNOWN_MINER_PROCESSES:
                if miner.lower() in proc_lower:
                    found.append(proc)
                    break
        return found

    @staticmethod
    def _check_suspicious_processes(process_names: list[str]) -> list[str]:
        """Check for processes that look suspicious (mining-adjacent)."""
        keywords = [
            "stratum", "mining", "miner", "pool.",
            "hashrate", "nicehash", "cryptonight",
            "ethash", "kawpow", "progpow",
        ]
        found = []
        for proc in process_names:
            proc_lower = proc.lower()
            for kw in keywords:
                if kw in proc_lower and proc not in found:
                    found.append(proc)
        return found

    @staticmethod
    def _get_container_cpu(container_id: str) -> float:
        """Get the current CPU percentage of a container."""
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", container_id],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode != 0:
                return 0.0
            cpu_str = result.stdout.strip().rstrip("%")
            return float(cpu_str) if cpu_str else 0.0
        except Exception:
            return 0.0

    def reset(self) -> None:
        """Clear CPU history (call when starting a new container)."""
        self._cpu_history.clear()


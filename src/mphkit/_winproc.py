"""
Processes and memory of Windows, for `mk.progress` and `mk.problem_size`,
read through the Windows API (ctypes): numbers, not text in the user's
language, and no package to install. Imported on Windows only.
"""
from __future__ import annotations

import sys

assert sys.platform == 'win32'

import ctypes
import mmap
import time
from ctypes import wintypes
from typing import Any

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
ntdll = ctypes.WinDLL('ntdll')

SNAPPROCESS = 0x2                 # CreateToolhelp32Snapshot: processes
QUERY_LIMITED = 0x1000            # PROCESS_QUERY_LIMITED_INFORMATION
ACCESS_DENIED = 5
STILL_ACTIVE = 259                # exit code of a running process
INVALID_HANDLE = ctypes.c_void_p(-1).value
LENGTH_MISMATCH = 0xC0000004      # STATUS_INFO_LENGTH_MISMATCH
PAGEFILE_INFORMATION = 18         # SystemPageFileInformation
EPOCH = 11644473600               # seconds from 1601 (FILETIME) to 1970
SAMPLE = 0.5                      # seconds over which CPU use is measured
CONSOLE_HOST = 'conhost.exe'      # the console of a console program


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD),
                ('cntUsage', wintypes.DWORD),
                ('th32ProcessID', wintypes.DWORD),
                ('th32DefaultHeapID', ctypes.c_void_p),
                ('th32ModuleID', wintypes.DWORD),
                ('cntThreads', wintypes.DWORD),
                ('th32ParentProcessID', wintypes.DWORD),
                ('pcPriClassBase', wintypes.LONG),
                ('dwFlags', wintypes.DWORD),
                ('szExeFile', wintypes.WCHAR*260)]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t)]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', wintypes.DWORD),
                ('dwMemoryLoad', wintypes.DWORD),
                ('ullTotalPhys', ctypes.c_uint64),
                ('ullAvailPhys', ctypes.c_uint64),
                ('ullTotalPageFile', ctypes.c_uint64),
                ('ullAvailPageFile', ctypes.c_uint64),
                ('ullTotalVirtual', ctypes.c_uint64),
                ('ullAvailVirtual', ctypes.c_uint64),
                ('ullAvailExtendedVirtual', ctypes.c_uint64)]


class SYSTEM_PAGEFILE_INFORMATION(ctypes.Structure):
    # the file name, a UNICODE_STRING, written out field by field
    _fields_ = [('NextEntryOffset', wintypes.ULONG),
                ('TotalSize', wintypes.ULONG),
                ('TotalInUse', wintypes.ULONG),
                ('PeakUsage', wintypes.ULONG),
                ('NameLength', wintypes.USHORT),
                ('NameMaximumLength', wintypes.USHORT),
                ('NameBuffer', ctypes.c_void_p)]


FILETIME = ctypes.POINTER(wintypes.FILETIME)
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
for _step in (kernel32.Process32FirstW, kernel32.Process32NextW):
    _step.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    _step.restype = wintypes.BOOL
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                 wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.GetProcessId.argtypes = [wintypes.HANDLE]
kernel32.GetProcessId.restype = wintypes.DWORD
kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, FILETIME, FILETIME,
                                     FILETIME, FILETIME]
kernel32.GetProcessTimes.restype = wintypes.BOOL
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                        ctypes.POINTER(wintypes.DWORD)]
kernel32.GetExitCodeProcess.restype = wintypes.BOOL
kernel32.K32GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
# unsigned, so that a status reads as written in the documentation
ntdll.NtQuerySystemInformation.argtypes = [
    ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong)]
ntdll.NtQuerySystemInformation.restype = ctypes.c_ulong


def snapshot() -> dict[int, tuple[int, str]] | None:
    """
    Returns the running processes, `{pid: (parent pid, program)}`, or
    `None` if Windows does not list them. An ended process is not listed,
    also while a handle keeps it; a parent pid is not updated when the
    parent ends (it may name a later process).
    """
    handle = kernel32.CreateToolhelp32Snapshot(SNAPPROCESS, 0)
    if handle is None or handle == INVALID_HANDLE:
        return None
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        rows = {}
        found = kernel32.Process32FirstW(handle, ctypes.byref(entry))
        while found:
            rows[entry.th32ProcessID] = (entry.th32ParentProcessID,
                                         entry.szExeFile)
            found = kernel32.Process32NextW(handle, ctypes.byref(entry))
        return rows
    finally:
        kernel32.CloseHandle(handle)


def facts(pid: int) -> dict[str, Any] | None:
    """
    Returns whether a process is `alive`, when it was `created` (seconds
    since 1970), its `cpu_s` (seconds of CPU so far) and `rss_mb` (working
    set), or `None` if there is no such process. A process of another
    user is alive with the rest `None`.
    """
    if not 0 < pid <= 0xFFFFFFFF:
        return None
    handle = kernel32.OpenProcess(QUERY_LIMITED, False, pid)
    if not handle:
        if ctypes.get_last_error() == ACCESS_DENIED:
            return {'alive': True, 'created': None, 'cpu_s': None,
                    'rss_mb': None}
        return None
    try:
        # Windows ignores a pid's lowest two bits: pid + 1 opens pid
        if kernel32.GetProcessId(handle) != pid:
            return None
        created, ended, kernel, user = (wintypes.FILETIME()
                                        for _ in range(4))
        code = wintypes.DWORD()
        if not (kernel32.GetProcessTimes(handle, ctypes.byref(created),
                                         ctypes.byref(ended),
                                         ctypes.byref(kernel),
                                         ctypes.byref(user))
                and kernel32.GetExitCodeProcess(handle, ctypes.byref(code))):
            return None
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        rss = (counters.WorkingSetSize // 2**20
               if kernel32.K32GetProcessMemoryInfo(
                   handle, ctypes.byref(counters), counters.cb) else None)
        return {'alive': code.value == STILL_ACTIVE,
                'created': _seconds(created) - EPOCH,
                'cpu_s': _seconds(kernel) + _seconds(user),
                'rss_mb': rss}
    finally:
        kernel32.CloseHandle(handle)


def _seconds(filetime: wintypes.FILETIME) -> float:
    """A FILETIME (100 ns steps) in seconds."""
    return ((filetime.dwHighDateTime << 32) | filetime.dwLowDateTime)/1e7


def processes(pid: int, started: float | None, written: float | None,
              slack: float) -> dict[str, Any]:
    """
    `mk.progress`'s process facts of Windows: whether `pid` is alive, and
    it and its descendants with their CPU use over half a second and
    memory. `started` (seconds since 1970) is the info file's start time
    and `written` the time the file was written, `None` with `pid=`.

    A process that started more than `slack` after `started` is another
    one with that pid. Windows keeps an ended parent's pid in its
    children, so the children of an ended (or reused) `pid` count only
    if they started between `started` and `written`; a COMSOL server
    started by the solve was there before `mk.log_progress` wrote the
    file. Processes that cannot be read (other users') are left out, as
    is the console host; the list is in the order the processes started.
    """
    first = snapshot()
    if first is None:
        return _without_list(pid, started, slack)
    alive, before = _members(pid, first, started, written, slack)
    if not before:
        return {'alive': alive, 'cpu_percent': None, 'rss_mb': None,
                'processes': []}
    start = time.perf_counter()
    time.sleep(SAMPLE)
    second = snapshot()
    if second is None:
        return _without_list(pid, started, slack)
    alive, after = _members(pid, second, started, written, slack)
    span = time.perf_counter() - start
    rows = []
    for member, info in after.items():
        program = second[member][1]
        if program.lower() == CONSOLE_HOST:
            continue
        old = before.get(member)
        if info['cpu_s'] is None:
            cpu = None
        elif (old is not None and old['cpu_s'] is not None
                and old['created'] == info['created']):
            cpu = round(100*(info['cpu_s'] - old['cpu_s'])/span, 1)
        else:                          # new: it used nothing before
            cpu = round(100*info['cpu_s']/span, 1)
        if program.lower().endswith('.exe'):
            program = program[:-4]
        rows.append((info['created'], member, program, cpu, info['rss_mb']))
    # unknown start times (another user's root) first
    rows.sort(key=lambda row: (row[0] is not None, row[0] or 0.0))
    cpus = [row[3] for row in rows]
    sizes = [row[4] for row in rows]
    return {'alive': alive,
            'cpu_percent': (None if None in cpus
                            else round(sum(c or 0.0 for c in cpus), 1)),
            'rss_mb': None if None in sizes else sum(s or 0 for s in sizes),
            'processes': [{'pid': member, 'command': program,
                           'cpu_percent': cpu, 'rss_mb': size}
                          for _, member, program, cpu, size in rows]}


def _members(pid: int, rows: dict[int, tuple[int, str]],
             started: float | None, written: float | None,
             slack: float) -> tuple[bool, dict[int, dict[str, Any]]]:
    """
    Returns whether `pid` is alive in `rows` and the facts of the live
    processes that count as its own (itself and its descendants).
    """
    own = facts(pid) if pid in rows else None
    reused = None
    if (own is not None and own['created'] is not None
            and started is not None and own['created'] > started + slack):
        reused, own = own['created'], None
    found: dict[int, dict[str, Any]] = {}
    # the start times a child of each member may have: (lowest, highest)
    limits: dict[int, tuple[float, float | None]] = {}
    if own is not None and own['alive']:
        found[pid] = own
        if own['created'] is not None:
            limits[pid] = (own['created'], None)
        elif started is not None:
            limits[pid] = (started - slack, None)
    elif started is not None and written is not None:
        latest = written + slack
        if reused is not None:
            latest = min(latest, reused)
        limits[pid] = (started - slack, latest)
    waiting = list(limits)
    while waiting:
        parent = waiting.pop()
        lowest, highest = limits[parent]
        for child, (parent_pid, _) in rows.items():
            if parent_pid != parent or child in found or child == pid:
                continue
            info = facts(child)
            if (info is None or not info['alive']
                    or info['created'] is None
                    or info['created'] < lowest
                    or (highest is not None and info['created'] > highest)):
                continue
            found[child] = info
            limits[child] = (info['created'], None)
            waiting.append(child)
    return pid in found, found


def _without_list(pid: int, started: float | None,
                  slack: float) -> dict[str, Any]:
    """The facts when Windows does not list the processes."""
    own = facts(pid)
    alive = bool(own and own['alive']
                 and not (own['created'] is not None and started is not None
                          and own['created'] > started + slack))
    return {'alive': alive, 'cpu_percent': None, 'rss_mb': None,
            'processes': None}


def memory() -> dict[str, int] | None:
    """Returns the computer's memory and the memory free for use, in MB."""
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {'total_mb': status.ullTotalPhys // 2**20,
            'free_mb': status.ullAvailPhys // 2**20}


def swap_used_mb() -> int | None:
    """Returns the page files' space in use, in MB."""
    size = 4096
    while size <= 65536:
        buffer = ctypes.create_string_buffer(size)
        length = ctypes.c_ulong()
        status = ntdll.NtQuerySystemInformation(
            PAGEFILE_INFORMATION, buffer, size, ctypes.byref(length))
        if status == LENGTH_MISMATCH:
            size *= 2
            continue
        if status != 0:
            return None
        pages, offset = 0, 0
        entry = ctypes.sizeof(SYSTEM_PAGEFILE_INFORMATION)
        while length.value >= offset + entry:   # none without a page file
            found = SYSTEM_PAGEFILE_INFORMATION.from_buffer(buffer, offset)
            pages += found.TotalInUse
            if not found.NextEntryOffset:
                break
            offset += found.NextEntryOffset
        return pages*mmap.PAGESIZE // 2**20
    return None

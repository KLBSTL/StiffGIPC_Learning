"""Windows owned process tree. No PID/name-wide termination or breakaway.

CreateProcessW(CREATE_SUSPENDED) -> AssignProcessToJobObject -> ResumeThread.
The non-inherited Job handle uses KILL_ON_JOB_CLOSE. Only explicitly selected
stdio handles are inherited. An assignment failure kills the still-suspended
process by its own handle. Query PID ownership comes from the Job, not ancestry
guesses or executable names. Source contracts:
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw
https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import time


class BASIC_LIMIT(C.Structure):
    _fields_ = [('PerProcessUserTimeLimit', C.c_int64), ('PerJobUserTimeLimit', C.c_int64),
                ('LimitFlags', W.DWORD), ('MinimumWorkingSetSize', C.c_size_t),
                ('MaximumWorkingSetSize', C.c_size_t), ('ActiveProcessLimit', W.DWORD),
                ('Affinity', C.c_size_t), ('PriorityClass', W.DWORD), ('SchedulingClass', W.DWORD)]


class IO_COUNTERS(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ('ReadOperationCount', 'WriteOperationCount',
                'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class EXTENDED_LIMIT(C.Structure):
    _fields_ = [('BasicLimitInformation', BASIC_LIMIT), ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', C.c_size_t), ('JobMemoryLimit', C.c_size_t),
                ('PeakProcessMemoryUsed', C.c_size_t), ('PeakJobMemoryUsed', C.c_size_t)]


class STARTUPINFO(C.Structure):
    _fields_ = [('cb', W.DWORD), ('lpReserved', W.LPWSTR), ('lpDesktop', W.LPWSTR),
                ('lpTitle', W.LPWSTR), ('dwX', W.DWORD), ('dwY', W.DWORD),
                ('dwXSize', W.DWORD), ('dwYSize', W.DWORD), ('dwXCountChars', W.DWORD),
                ('dwYCountChars', W.DWORD), ('dwFillAttribute', W.DWORD),
                ('dwFlags', W.DWORD), ('wShowWindow', W.WORD), ('cbReserved2', W.WORD),
                ('lpReserved2', C.POINTER(C.c_ubyte)), ('hStdInput', W.HANDLE),
                ('hStdOutput', W.HANDLE), ('hStdError', W.HANDLE)]


class STARTUPINFOEX(C.Structure):
    _fields_ = [('StartupInfo', STARTUPINFO), ('lpAttributeList', W.LPVOID)]


class PROCESS_INFORMATION(C.Structure):
    _fields_ = [('hProcess', W.HANDLE), ('hThread', W.HANDLE),
                ('dwProcessId', W.DWORD), ('dwThreadId', W.DWORD)]


def _api():
    if os.name != 'nt':
        raise OSError('Windows Job Object requires Windows')
    api = C.WinDLL('kernel32', use_last_error=True)
    declarations = {
        'CreateJobObjectW': ([W.LPVOID, W.LPCWSTR], W.HANDLE),
        'SetInformationJobObject': ([W.HANDLE, C.c_int, W.LPVOID, W.DWORD], W.BOOL),
        'QueryInformationJobObject': ([W.HANDLE, C.c_int, W.LPVOID, W.DWORD, C.POINTER(W.DWORD)], W.BOOL),
        'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
        'IsProcessInJob': ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
        'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL),
        'CloseHandle': ([W.HANDLE], W.BOOL),
        'CreateProcessW': ([W.LPCWSTR, W.LPWSTR, W.LPVOID, W.LPVOID, W.BOOL,
                           W.DWORD, W.LPVOID, W.LPCWSTR, C.POINTER(STARTUPINFOEX),
                           C.POINTER(PROCESS_INFORMATION)], W.BOOL),
        'ResumeThread': ([W.HANDLE], W.DWORD),
        'TerminateProcess': ([W.HANDLE, W.UINT], W.BOOL),
        'WaitForSingleObject': ([W.HANDLE, W.DWORD], W.DWORD),
        'GetExitCodeProcess': ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL),
        'InitializeProcThreadAttributeList': ([W.LPVOID, W.DWORD, W.DWORD, C.POINTER(C.c_size_t)], W.BOOL),
        'UpdateProcThreadAttribute': ([W.LPVOID, W.DWORD, C.c_size_t, W.LPVOID,
                                      C.c_size_t, W.LPVOID, W.LPVOID], W.BOOL),
        'DeleteProcThreadAttributeList': ([W.LPVOID], None),
        'OpenProcess': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
        'QueryFullProcessImageNameW': ([W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)], W.BOOL),
        'GetProcessTimes': ([W.HANDLE, C.POINTER(W.FILETIME), C.POINTER(W.FILETIME),
                             C.POINTER(W.FILETIME), C.POINTER(W.FILETIME)], W.BOOL),
    }
    for name, (args, result) in declarations.items():
        fn = getattr(api, name); fn.argtypes = args; fn.restype = result
    return api


def _check(ok, name):
    if not ok:
        raise C.WinError(C.get_last_error(), name)


class OwnedJob:
    def __init__(self):
        self.api = _api()
        self.job = self.api.CreateJobObjectW(None, None)
        _check(self.job, 'CreateJobObjectW')
        self.process = None
        self.pid = None
        self.lifecycle = []
        self.members = {}
        limits = EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE; no breakaway.
        try:
            _check(self.api.SetInformationJobObject(self.job, 9, C.byref(limits), C.sizeof(limits)),
                   'SetInformationJobObject')
        except BaseException:
            self.api.CloseHandle(self.job); self.job = None
            raise

    def _assign(self, process):
        _check(self.api.AssignProcessToJobObject(self.job, process), 'AssignProcessToJobObject')

    def launch(self, command, cwd, env, log):
        import msvcrt
        if self.process is not None:
            raise RuntimeError('Only one root launch per owned job')
        executable = str(Path(command[0]).resolve(strict=True))
        if any('\0' in str(s) for s in command):
            raise ValueError('NUL in command')
        pairs = []
        for key, value in env.items():
            if not key or '=' in key or '\0' in key or '\0' in value:
                raise ValueError('Invalid environment entry')
            pairs.append((key, value))
        block = C.create_unicode_buffer('\0'.join(k+'='+v for k,v in sorted(pairs, key=lambda kv: kv[0].upper()))+'\0\0')
        cmd = C.create_unicode_buffer(subprocess.list2cmdline([executable] + [str(s) for s in command[1:]]))
        info = PROCESS_INFORMATION()
        size = C.c_size_t()
        self.api.InitializeProcThreadAttributeList(None, 1, 0, C.byref(size))
        if not size.value:
            raise C.WinError(C.get_last_error(), 'Attribute list size')
        attrs = C.create_string_buffer(size.value)
        _check(self.api.InitializeProcThreadAttributeList(attrs, 1, 0, C.byref(size)), 'Initialize attributes')
        try:
            with open(os.devnull, 'rb') as null:
                handles = [msvcrt.get_osfhandle(null.fileno()), msvcrt.get_osfhandle(log.fileno())]
                old = [os.get_handle_inheritable(h) for h in handles]
                try:
                    for h in handles: os.set_handle_inheritable(h, True)
                    selected = (W.HANDLE * len(handles))(*handles)
                    _check(self.api.UpdateProcThreadAttribute(attrs, 0, 0x20002, selected,
                           C.sizeof(selected), None, None), 'Handle inheritance whitelist')
                    startup = STARTUPINFOEX()
                    startup.StartupInfo.cb = C.sizeof(startup)
                    startup.StartupInfo.dwFlags = 0x100 | 1  # USESTDHANDLES | USESHOWWINDOW.
                    startup.StartupInfo.wShowWindow = 0
                    startup.StartupInfo.hStdInput = handles[0]
                    startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = handles[1]
                    startup.lpAttributeList = C.cast(attrs, W.LPVOID)
                    flags = 0x4 | 0x400 | 0x80000 | 0x08000000
                    _check(self.api.CreateProcessW(executable, cmd, None, None, True, flags,
                           block, str(Path(cwd).resolve()), C.byref(startup), C.byref(info)), 'CreateProcessW suspended')
                finally:
                    for h, state in zip(handles, old): os.set_handle_inheritable(h, state)
            self.lifecycle.append('created_suspended')
            self._assign(info.hProcess)
            self.lifecycle.append('assigned_to_job')
            if self.api.ResumeThread(info.hThread) == 0xffffffff:
                raise C.WinError(C.get_last_error(), 'ResumeThread')
            self.lifecycle.append('resumed')
            self.process, self.pid = info.hProcess, int(info.dwProcessId)
        except BaseException:
            # Covers every post-CreateProcess failure, including restoration of
            # temporary stdio inheritance before assignment. The exact process
            # handle remains ours; it cannot have run before a successful resume.
            if info.hProcess and self.process is None:
                if self.exit_code(info.hProcess) is None:
                    _check(self.api.TerminateProcess(info.hProcess, 1), 'Terminate failed launch')
                state = self.api.WaitForSingleObject(info.hProcess, 5000)
                if state != 0:
                    raise RuntimeError('Failed-launch process did not stop within cleanup budget')
                self.api.CloseHandle(info.hProcess)
            raise
        finally:
            if info.hThread:
                self.api.CloseHandle(info.hThread)
            self.api.DeleteProcThreadAttributeList(attrs)
        return self.pid

    def pids(self):
        capacity = 32
        while capacity <= 65536:
            class PID_LIST(C.Structure):
                _fields_ = [('assigned', W.DWORD), ('count', W.DWORD), ('pids', C.c_size_t * capacity)]
            data = PID_LIST(); returned = W.DWORD()
            ok = self.api.QueryInformationJobObject(self.job, 3, C.byref(data), C.sizeof(data), C.byref(returned))
            if ok and data.count >= data.assigned:
                return {int(p) for p in data.pids[:data.count]}
            if not ok and C.get_last_error() != 234:
                raise C.WinError(C.get_last_error(), 'Query Job process IDs')
            capacity = max(capacity * 2, int(data.assigned))
        raise RuntimeError('Owned job PID capacity exceeded')

    def owns_pid(self, pid):
        # Re-check current kernel membership, avoiding stale observed PID sets.
        handle = self.api.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            inside = W.BOOL()
            _check(self.api.IsProcessInJob(handle, self.job, C.byref(inside)), 'IsProcessInJob')
            return bool(inside.value)
        finally:
            self.api.CloseHandle(handle)

    def observe(self):
        for pid in self.pids():
            handle = self.api.OpenProcess(0x1000 | 0x100000, False, pid)
            if not handle:  # The member can exit between queries.
                continue
            retained = False
            try:
                inside = W.BOOL()
                _check(self.api.IsProcessInJob(handle, self.job, C.byref(inside)), 'Member identity')
                if not inside.value: continue
                times = [W.FILETIME() for _ in range(4)]
                _check(self.api.GetProcessTimes(handle, *(C.byref(t) for t in times)), 'GetProcessTimes')
                created = (int(times[0].dwHighDateTime) << 32) | int(times[0].dwLowDateTime)
                identity = (pid, created)
                if identity in self.members: continue
                text = C.create_unicode_buffer(32768); size = W.DWORD(len(text))
                image_ok = self.api.QueryFullProcessImageNameW(handle, 0, text, C.byref(size))
                image_error = 0 if image_ok else C.get_last_error()
                if not image_ok and self.exit_code(handle) is None:
                    # A live member whose identity cannot be inspected still
                    # fails closed. Only a signalled exact process handle can
                    # explain this race; PID/name guesses are never sufficient.
                    raise C.WinError(image_error, 'Member image')
                self.members[identity] = {'pid': pid, 'creation_time_100ns': created,
                                          'image': text.value if image_ok else None,
                                          'image_query_error': image_error,
                                          'exited_before_image_query': not bool(image_ok),
                                          '_handle': handle}
                retained = True
            finally:
                if not retained: self.api.CloseHandle(handle)
        return self.member_evidence()

    def exit_code(self, handle):
        state = self.api.WaitForSingleObject(handle, 0)
        if state == 258: return None
        if state != 0: raise C.WinError(C.get_last_error(), 'WaitForSingleObject')
        code = W.DWORD()
        _check(self.api.GetExitCodeProcess(handle, C.byref(code)), 'GetExitCodeProcess')
        return int(code.value)

    def poll(self):
        return self.exit_code(self.process) if self.process else None

    def member_evidence(self):
        return [{k:v for k,v in row.items() if k != '_handle'} |
                {'exit_code': self.exit_code(row['_handle'])} for row in self.members.values()]

    def finished(self):
        # Windows can remove the final PID from a Job before its process handle
        # is signalled. Tree emptiness alone is insufficient exit evidence.
        return (not self.pids() and (not self.process or self.poll() is not None)
                and all(self.exit_code(r['_handle']) is not None for r in self.members.values()))

    def terminate(self, timeout=5):
        _check(self.api.TerminateJobObject(self.job, 1), 'Terminate owned Job')
        until = time.monotonic() + timeout
        while not self.finished():
            if time.monotonic() >= until:
                raise TimeoutError('Owned Job/process handles did not finish after termination')
            time.sleep(.02)

    def close(self):
        if self.job:
            try:
                if not self.finished(): self.terminate()
            finally:
                self.api.CloseHandle(self.job); self.job = None
                for member in self.members.values(): self.api.CloseHandle(member['_handle'])
                self.members.clear()
                if self.process: self.api.CloseHandle(self.process); self.process = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

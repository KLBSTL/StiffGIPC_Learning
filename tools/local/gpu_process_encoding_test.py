"""Process-name decoding must preserve the foreign-load resource gate."""
import unittest
from windows_runner import gpu_process_text,process_rows


class ProcessEncodingTests(unittest.TestCase):
    def test_utf8(self):
        text='12, C:/程序/app.exe, N/A\n'
        self.assertEqual(gpu_process_text(text.encode('utf-8'),'cp936'),text)

    def test_windows_ansi(self):
        text='12, C:/程序/app.exe, N/A\n'
        self.assertEqual(gpu_process_text(text.encode('cp936'),'cp936'),text)

    def test_foreign_gate_unchanged(self):
        text='12, C:/程序/gipc.exe, N/A\n13, C:/程序/app.exe, 32\n'
        rows=process_rows(gpu_process_text(text.encode('cp936'),'cp936'),99,'WDDM')
        self.assertEqual(rows['blocking_foreign_pids'],[12,13])

    def test_invalid_bytes_are_rejected(self):
        with self.assertRaises(UnicodeDecodeError):gpu_process_text(b'\xff','cp936')

    def test_empty(self):
        self.assertEqual(gpu_process_text(b'','cp936'),'')


if __name__=='__main__':unittest.main()

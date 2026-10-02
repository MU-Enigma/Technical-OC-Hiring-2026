#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.auth import destroy_session

def main():
    cookie_header = destroy_session()
    print("Status: 302 Found")
    print(cookie_header)
    print("Location: /cgi-bin/login.py")
    print()

if __name__ == "__main__":
    main()

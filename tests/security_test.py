"""
Unit Test - Testing units in `security.py`

Author:
-------
Anirudh K.M Kakarlapudi
"""
from einstein.utils.security import Encryption, Decryption
import os
import shutil

encr = Encryption()
decr = Decryption()


def test_generate_key():
    os.mkdir('temp')
    gen_key = encr.generate_key('key_test.txt')
    shutil.rmtree('temp')
    assert (len(encr.key) > 0)


def test_load_key():
    os.mkdir('temp')
    load_key = decr.load_key('key.txt')
    shutil.rmtree('temp')
    assert (len(decr.key) > 0)

import pyAesCrypt
import binascii
import os
import subprocess
import shutil


class Security:
    def __init__(self, key_path=KEY_PATH, file_path=DATA_PATH,
                 encrypt_path=ENCRYPTED_PATH, decrypt_path=DECRYPTED_PATH):
        self.bufferSize = 64 * 1024
        self.key_path = key_path
        self.encrypt_path = encrypt_path
        self.decrypt_path = decrypt_path
        self.file_path = file_path

    def generate_key(self, key_file_name):
        os.mkdir('key')
        key = binascii.b2a_hex(os.urandom(15))
        key_text = key.decode()
        with open('key/'+key_file_name, 'w') as file:
            file.write(key_text)
        self.upload('key', self.key_path)
        shutil.rmtree('key')
        return key_text

    def load_key(self, key_file_name):
        os.mkdir('key')
        self.download('key', self.key_path)
        with open('key/'+key_file_name, 'r') as file:
            gen_key = file.read()
        return gen_key

    def encrypt(self):
        # download all files locally
        os.mkdir('data')
        self.download('data', self.file_path)
        # print('--------->')
        os.mkdir('encrypt')
        self.key = self.generate_key('key.txt')
        for file in os.listdir('data/'):
            pyAesCrypt.encryptFile('data/'+file, 'encrypt/'+file+".aes",
                                   self.key, self.bufferSize)
        self.upload('encrypt', self.encrypt_path)
        shutil.rmtree('encrypt')
        shutil.rmtree('data')

    def download(self, foldername, gspath):
        print(f'gsutil -m cp -r {gspath}* {foldername}/')
        subprocess.run(f'gsutil -m cp -r {gspath}* {foldername}/', shell=True)

    def upload(self, foldername, gspath):
        print(f'gsutil cp -r {foldername}/* {gspath}')
        subprocess.run(f'gsutil cp -r {foldername}/* {gspath}', shell=True)

    def decrypt(self):
        os.mkdir('encrypt')
        self.download('encrypt', self.encrypt_path)
        os.mkdir('decrypt')
        key = self.load_key('key.txt')
        for file in os.listdir('encrypt/'):
            pyAesCrypt.decryptFile('encrypt/'+file, 'decrypt/'+file[:-4],
                                   key, self.bufferSize)
        self.upload('decrypt', self.decrypt_path)
        shutil.rmtree('decrypt')
        shutil.rmtree('encrypt')
        shutil.rmtree('key')

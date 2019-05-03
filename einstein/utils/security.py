"""
This script implements the :class: `Encryption`, and :class: `Decryption`
which encrypt and decrypt the data files.

Author:
----------
Anirudh K.M. Kakarlapudi
"""
import os
import shutil
import binascii
import subprocess
import pyAesCrypt


class Encryption:
    """A class for encrypting the data files
    """
    def __init__(self, key_path='gs://dsp_uga/',
                 file_path='gs://uga_dsp_sp19/',
                 encrypt_path='gs://profinal/'):
        """Initialises the class.

        Args:
            key_path(str):
                A google storage path where the key should be stored
            file_path(str):
                A google storage path where the data files are stored
            encrypt_path(str):
                A google storage path where the enrypted files should be stored
        """
        self.buffer_size = 64 * 1024
        self.key_path = key_path
        self.encrypt_path = encrypt_path
        self.file_path = file_path

    def generate_key(self, key_file_name):
        """Function to generate the random 16 digit key and store in the
        storage bucket

        Args:
            key_file_name(str):
                The name of the key file
        Returns:
            The 16 digit key
        """
        key = binascii.b2a_hex(os.urandom(15))
        self.key = key.decode()
        with open('temp/'+key_file_name, 'w') as file:
            file.write(self.key)
        self.upload('key.txt', self.key_path)

    def encrypt(self):
        """Function to encrypt all the files in the bucket.
        Uploads all the files into the bucket and deletes
        all the files.
        """
        os.mkdir('temp')
        self.download(self.file_path)
        self.generate_key('key.txt')
        files = []
        files += [file for file in os.listdir('temp/')
                  if file.endswith('.csv')]
        for file in files:
            pyAesCrypt.encryptFile('temp/' + file, 'temp/' + file + ".aes",
                                   self.key, self.buffer_size)
            self.upload(file + ".aes", self.encrypt_path)
        shutil.rmtree('temp')

    def download(self, gspath):
        """Function to download all the files in the storage bucket

        Args:
            gspath(str):
                 A google storage path where the enrypted files should be
                 stored
        """
        subprocess.run(f'gsutil rsync -r {gspath} ~/einstein/temp/', shell=True)

    def upload(self, file_name, gspath):
        """Function to upload the file into the storage bucket
        Args:
            file_name(str):
                The name of file to be uploaded
        """
        subprocess.run(f'gsutil -m cp ~/einstein/temp/{file_name} {gspath}', shell=True)


class Decryption:
    """A class for decryting the data files
    """
    def __init__(self, key_path,
                 encrypt_path,
                 decrypt_path):
        """Initialises the class.

        Args:
            key_path(str):
                A google storage path where the key should be stored
            decrypt_path(str):
                A google storage path where the decrypt files should be stored
            encrypt_path(str):
                A google storage path where the enrypted files are stored
        """
        self.buffer_size = 64 * 1024
        self.key_path = key_path
        self.encrypt_path = encrypt_path
        self.decrypt_path = decrypt_path

    def load_key(self, key_file_name):
        """Function to generate the random 16 digit key and store in the
        storage bucket

        Args:
            key_file_name(str):
                The name of the key file
        Returns:
            The 16 digit key
        """
        self.download(key_file_name, self.key_path)
        with open('temp/'+key_file_name, 'r') as file:
            self.key = file.read()

    def decrypt(self, file_name):
        """Function to encrypt all the files in the bucket.
        Uploads all the files into the bucket.

        Args:
            file_name(str):
                The name of file to be decrypted
        """
        os.mkdir('temp')
        self.download(file_name + ".aes", self.encrypt_path)
        self.load_key('key.txt')
        pyAesCrypt.decryptFile('temp/' + file_name + ".aes", 'temp/' + file_name,
                               self.key, self.buffer_size)
        self.upload(file_name, self.decrypt_path)
        shutil.rmtree('temp')

    def delete_data(self, file_name, gspath):
        """Deletes all the files in the temp folder

        Args:
            file_name(str):
                The name of the file to be removed from google storage bucket
            gspath(str):
                The google storage bucket path from where the file should be
                deleted
        """
        subprocess.run(f'gsutil rm {gspath}{file_name}', shell=True)

    def download(self, file_name, gspath):
        """Downloads the file in the storage bucket

        Args:
            file_name(str):
                The name of the file to be downloaded from google storage
                bucket
            gspath(str):
                The google storage bucket path from where the file should be
                downloaded
        """
        subprocess.run(f'gsutil -m cp {gspath}{file_name} ~/einstein/temp/', shell=True)

    def upload(self, file_name, gspath):
        """Uploads the file in the storage bucket

        Args:
            file_name(str):
                The name of the file to be uploaded from google storage bucket
            gspath(str):
                The google storage bucket path from where the encrypt file
                should be uploaded
        """
        subprocess.run(f'gsutil -m cp ~/einstein/temp/{file_name} {gspath}', shell=True)

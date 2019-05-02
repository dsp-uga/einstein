"""
This script implements the :class: Encryption, and :class: Decryption
which encrypt and decrypt the data files.

Author:
----------
Anirudh K.M. Kakarlapudi
"""
import pyAesCrypt
import binascii
import os
import subprocess
import shutil


class Encryption:
    """A class for encrypting the data files
    """
    def __init__(self, key_path='gs://dsp_uga/',
                 file_path='gs://uga_dsp_sp19/',
                 encrypt_path='gs://profinal/'):
        """Initialises the class.

        Args:
            key_path(path):
                A google storage path where the key should be stored
            file_path(path):
                A google storage path where the data files are stored
            encrypt_path(path):
                A google storage path where the enrypted files should be stored
        """
        self.bufferSize = 64 * 1024
        self.key_path = key_path
        self.encrypt_path = encrypt_path
        self.file_path = file_path

    def generate_key(self, key_file_name):
        """Function to generate the random 16 digit key and store in the
        storage bucket

        Args:
            key_file_name:
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
                                   self.key, self.bufferSize)
            self.upload(file + ".aes", self.encrypt_path)
        shutil.rmtree('temp')

    def download(self, gspath):
        """Function to download all the files in the storage bucket
        """
        subprocess.run(f'gsutil rsync -r {gspath} ~/temp/', shell=True)

    def upload(self, filename, gspath):
        """Function to upload the file into the storage bucket
        """
        subprocess.run(f'gsutil -m cp ~/temp/{filename} {gspath}', shell=True)


class Decryption:
    """A class for decryting the data files
    """
    def __init__(self, key_path='gs://dsp_uga/',
                 decrypt_path='gs://dsp_uga/',
                 encrypt_path='gs://profinal/'):
        """Initialises the class.

        Args:
            key_path(path):
                A google storage path where the key should be stored
            decrypt_path(path):
                A google storage path where the decrypt files should be stored
            encrypt_path(path):
                A google storage path where the enrypted files are stored
        """
        self.bufferSize = 64 * 1024
        self.key_path = key_path
        self.encrypt_path = encrypt_path
        self.decrypt_path = decrypt_path

    def load_key(self, key_file_name):
        """Function to generate the random 16 digit key and store in the
        storage bucket

        Args:
            key_file_name:
                    The name of the key file
        Returns:
            The 16 digit key
        """
        self.download('key.txt', self.key_path)
        with open('temp/'+key_file_name, 'r') as file:
            self.key = file.read()

    def decrypt(self, filename):
        """Function to encrypt all the files in the bucket.
        Uploads all the files into the bucket.
        """
        os.mkdir('temp')
        self.download(filename + ".aes", self.encrypt_path)
        self.load_key('key.txt')
        pyAesCrypt.decryptFile('temp/' + filename + ".aes", 'temp/' + filename,
                               self.key, self.bufferSize)
        self.upload(filename, self.decrypt_path)
        shutil.rmtree('temp')

    def delete_data(self, filename, gspath):
        """Function to delete all the files in the temp folder

        Args:
            filename:
                The name of the file to be removed from google storage bucket
            gspath:
                The google storage bucket path from where the file should be
                deleted
        """
        subprocess.run(f'gsutil rm {gspath}{filename}', shell=True)

    def download(self, filename, gspath):
        """Function to download the file in the storage bucket

        Args:
            filename:
                The name of the file to be downloaded from google storage
                bucket
            gspath:
                The google storage bucket path from where the file should be
                downloaded
        """
        subprocess.run(f'gsutil -m cp {gspath}{filename} ~/temp/', shell=True)

    def upload(self, filename, gspath):
        """Function to upload the file in the storage bucket

        Args:
            filename:
                The name of the file to be uploaded from google storage bucket
            gspath:
                The google storage bucket path from where the encrypt file
                should be uploaded
        """
        subprocess.run(f'gsutil -m cp ~/temp/{filename} {gspath}', shell=True)

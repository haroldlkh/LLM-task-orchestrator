import io
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

class GDriveConnector:
    def __init__(self, service_account_json):
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.creds = service_account.Credentials.from_service_account_info(
            service_account_json, scopes=self.scopes
        )
        self.service = build('drive', 'v3', credentials=self.creds)

    def list_files_in_folder(self, folder_id):
        """Returns a list of all parquet files in a folder."""
        query = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        return results.get('files', [])

    def download_file(self, file_id, destination):
        request = self.service.files().get_media(fileId=file_id)
        with io.FileIO(destination, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

    # def upload_file(self, local_path, folder_id, remote_name, owner_email):
    #     file_metadata = {
    #         'name': remote_name,
    #         'parents': [folder_id]
    #     }
    #     media = MediaFileUpload(local_path, resumable=True)
        
    #     # 1. Create the file (initially owned by Service Account)
    #     file = self.service.files().create(
    #         body=file_metadata,
    #         media_body=media,
    #         fields='id',
    #         supportsAllDrives=True
    #     ).execute()
        
    #     file_id = file.get('id')

    #     # 2. Transfer Ownership to your personal email to use your quota
    #     permission = {
    #         'type': 'user',
    #         'role': 'owner',
    #         'emailAddress': owner_email
    #     }
        
    #     self.service.permissions().create(
    #         fileId=file_id,
    #         body=permission,
    #         transferOwnership=True,
    #         supportsAllDrives=True
    #     ).execute()

    #     return file_id
    
    def upload_file(self, local_path, folder_id, remote_name, owner_email=None):
            file_metadata = {
                'name': remote_name,
                'parents': [folder_id]
            }
            
            # KEY CHANGE: resumable=False
            # Simple upload mode doesn't check 'session' quota
            media = MediaFileUpload(local_path, resumable=False)
            
            print(f"Attempting simple upload of {remote_name} to folder {folder_id}...")
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
            
            return file.get('id')

    #simple create, then reassign
    # def upload_file(self, local_path, folder_id, remote_name, owner_email):
    #         file_metadata = {
    #             'name': remote_name,
    #             'parents': [folder_id]
    #         }
            
    #         # 1. Create a "Stub" (Empty File) using a Simple Upload
    #         # Simple uploads (resumable=False) usually bypass the 0GB quota check
    #         empty_media = MediaFileUpload(local_path, resumable=False) 
    #         file = self.service.files().create(
    #             body=file_metadata,
    #             media_body=empty_media,
    #             fields='id',
    #             supportsAllDrives=True
    #         ).execute()
    #         file_id = file.get('id')

    #         # 2. Transfer Ownership immediately
    #         # Now YOU own the file ID, and the quota is YOURS.
    #         permission = {'type': 'user', 'role': 'owner', 'emailAddress': owner_email}
    #         self.service.permissions().create(
    #             fileId=file_id, 
    #             body=permission, 
    #             transferOwnership=True, 
    #             supportsAllDrives=True
    #         ).execute()

    #         # 3. NOW perform the Resumable Upload into the existing ID
    #         # Since you own the file now, this will use your quota.
    #         resumable_media = MediaFileUpload(local_path, resumable=True)
    #         self.service.files().update(
    #             fileId=file_id,
    #             media_body=resumable_media,
    #             supportsAllDrives=True
    #         ).execute()

    #         return file_id
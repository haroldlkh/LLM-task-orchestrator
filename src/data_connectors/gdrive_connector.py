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
    
    def upload_file(self, local_path, folder_id, remote_name, owner_email):
        file_metadata = {
            'name': remote_name,
            'parents': [folder_id]
        }
        
        # Resumable=True is what we want for scalability
        media = MediaFileUpload(local_path, resumable=True)

        # 1. Create the initial file entry
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True
        ).execute()
        file_id = file.get('id')

        # 2. IMMEDIATELY transfer ownership to your personal email
        # This shifts the storage "bill" from the Service Account to YOU
        try:
            permission = {
                'type': 'user',
                'role': 'owner',
                'emailAddress': owner_email
            }
            self.service.permissions().create(
                fileId=file_id,
                body=permission,
                transferOwnership=True,
                supportsAllDrives=True
            ).execute()
        except Exception as e:
            # If ownership transfer fails, we still have the file, 
            # but it might hit quota later.
            print(f"Ownership transfer warning: {e}")

        return file_id
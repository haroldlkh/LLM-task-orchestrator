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

    def upload_file(self, local_path, folder_id, filename):
        file_metadata = {'name': filename, 'parents': [folder_id]}
        media = MediaFileUpload(local_path, resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
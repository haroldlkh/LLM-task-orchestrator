import io
import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload


class GDriveConnector:
    def __init__(self):
        self.scopes = ["https://www.googleapis.com/auth/drive"]

        self.creds = Credentials(
            token=None,
            refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["GDRIVE_CLIENT_ID"],
            client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
            scopes=self.scopes,
        )

        self.creds.refresh(Request())
        self.service = build("drive", "v3", credentials=self.creds)

    def list_files_in_folder(self, folder_id):
        query = (
            f"'{folder_id}' in parents "
            f"and trashed = false "
            f"and mimeType != 'application/vnd.google-apps.folder'"
        )

        files = []
        page_token = None

        while True:
            results = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                pageToken=page_token,
            ).execute()

            files.extend(results.get("files", []))
            page_token = results.get("nextPageToken")

            if not page_token:
                break

        return files

    def download_file(self, file_id, destination):
        request = self.service.files().get_media(fileId=file_id)

        with io.FileIO(destination, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def upload_file(self, local_path, folder_id, remote_name):
        file_metadata = {
            "name": remote_name,
            "parents": [folder_id],
        }

        media = MediaFileUpload(local_path, resumable=True)

        result = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name",
            supportsAllDrives=True,
        ).execute()

        return result["id"]
import io
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from .base import BaseConnector


class GDriveConnector(BaseConnector):
    """
    Supported auth modes via config JSON:

    1) oauth_user
       For personal Google accounts or any case where uploads should use
       a human user's Drive quota.

    2) service_account
       For server-to-server auth as the service account itself.
       Suitable for shared-drive workflows, not personal My Drive uploads.

    3) service_account_delegated
       For Google Workspace with domain-wide delegation.
       Service account impersonates a Workspace user via delegated_subject.
    """

    DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive"]
    DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"

    def __init__(self, config):
        super().__init__(config)

        self.auth_type = self.config.get("auth_type", "oauth_user").strip()
        self.scopes = self.config.get("scopes", self.DEFAULT_SCOPES)

        self.creds = self._build_credentials()
        self.service = build("drive", "v3", credentials=self.creds)

    def _build_credentials(self):
        if self.auth_type == "oauth_user":
            return self._build_oauth_user_credentials()

        if self.auth_type == "service_account":
            return self._build_service_account_credentials()

        if self.auth_type == "service_account_delegated":
            return self._build_service_account_delegated_credentials()

        raise ValueError(
            f"Unsupported GDrive auth_type '{self.auth_type}'. "
            f"Supported values: oauth_user, service_account, service_account_delegated"
        )

    def _build_oauth_user_credentials(self):
        required = ["client_id", "client_secret", "refresh_token"]
        missing = [k for k in required if not self.config.get(k)]
        if missing:
            raise ValueError(
                f"GDrive oauth_user config missing required keys: {missing}"
            )

        token_uri = self.config.get("token_uri", self.DEFAULT_TOKEN_URI)

        creds = UserCredentials(
            token=None,
            refresh_token=self.config["refresh_token"],
            token_uri=token_uri,
            client_id=self.config["client_id"],
            client_secret=self.config["client_secret"],
            scopes=self.scopes,
        )

        creds.refresh(Request())
        return creds

    def _build_service_account_credentials(self):
        service_account_info = self.config.get("service_account_info")
        if not service_account_info:
            raise ValueError(
                "GDrive service_account config requires 'service_account_info'"
            )

        creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=self.scopes,
        )
        return creds

    def _build_service_account_delegated_credentials(self):
        service_account_info = self.config.get("service_account_info")
        delegated_subject = self.config.get("delegated_subject")

        if not service_account_info:
            raise ValueError(
                "GDrive service_account_delegated config requires 'service_account_info'"
            )
        if not delegated_subject:
            raise ValueError(
                "GDrive service_account_delegated config requires 'delegated_subject'"
            )

        creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=self.scopes,
        ).with_subject(delegated_subject)

        return creds

    def list_objects(self, location: str):
        query = (
            f"'{location}' in parents "
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

    def download_object(self, object_id: str, destination: str):
        request = self.service.files().get_media(fileId=object_id)

        with io.FileIO(destination, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def upload_object(self, local_path: str, destination: str, remote_name: str):
        file_metadata = {
            "name": remote_name,
            "parents": [destination],
        }

        media = MediaFileUpload(local_path, resumable=True)

        result = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name",
            supportsAllDrives=True,
        ).execute()

        return result["id"]
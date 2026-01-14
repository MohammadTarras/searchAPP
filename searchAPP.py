import streamlit as st
import os
import io
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import pickle
import numpy as np
import base64
import streamlit.components.v1 as components
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json

class GoogleDriveImageSearchEngine:
    def __init__(self, credentials_dict, folder_id):
        """
        Initialize with Google Drive credentials and folder ID
        credentials_dict: Dictionary containing service account credentials
        folder_id: Google Drive folder ID to search in
        """
        self.folder_id = folder_id
        self.image_features = {}
        self.image_metadata = {}  # Store file IDs and names
        self.is_indexed = False
        
        # Initialize Google Drive API
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        self.drive_service = build('drive', 'v3', credentials=credentials)

    def list_images_in_folder(self):
        """List all images in the specified Google Drive folder"""
        query = f"'{self.folder_id}' in parents and (mimeType contains 'image/')"
        results = self.drive_service.files().list(
            q=query,
            fields="files(id, name, mimeType)",
            pageSize=1000
        ).execute()
        return results.get('files', [])

    def download_image(self, file_id):
        """Download an image from Google Drive"""
        request = self.drive_service.files().get_media(fileId=file_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        file_buffer.seek(0)
        return file_buffer

    def preprocess_image_from_buffer(self, image_buffer):
        """Preprocess image from buffer"""
        img_array = np.frombuffer(image_buffer.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))
        return resized

    def preprocess_pil_image(self, pil_image):
        """Preprocess PIL Image (from clipboard)"""
        img_array = np.array(pil_image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img = img_array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        resized = cv2.resize(gray, (256, 256))
        return resized
    
    
    def build_index(self):
        """Index only images that are not already indexed"""
        
        cache_file = "gdrive_index_cache.pkl"
        
        # Load cached index if exists
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.image_features = cached_data.get('features', {})
                self.image_metadata = cached_data.get('metadata', {})
        else:
            self.image_features = {}
            self.image_metadata = {}

        files = self.list_images_in_folder()
        if not files:
            st.info("No images found in the folder.")
            return
        
        # Filter files that are already indexed
        new_files = [f for f in files if f['id'] not in self.image_metadata]
        if not new_files:
            st.success("All images are already indexed!")
            self.is_indexed = True
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file in enumerate(new_files):
            try:
                status_text.text(f"Processing {file['name']} ({idx+1}/{len(new_files)})")
                
                # Download and process image
                image_buffer = self.download_image(file['id'])
                processed_img = self.preprocess_image_from_buffer(image_buffer)
                
                # Store features and metadata
                self.image_features[file['id']] = processed_img
                self.image_metadata[file['id']] = {
                    'name': file['name'],
                    'mimeType': file['mimeType']
                }

                progress_bar.progress((idx + 1) / len(new_files))
            
            except Exception as e:
                st.warning(f"Error processing {file['name']}: {e}")
        
        # Update cache
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'features': self.image_features,
                'metadata': self.image_metadata
            }, f)

        progress_bar.empty()
        status_text.empty()
        st.success(f"Indexed {len(new_files)} new images successfully!")
        self.is_indexed = True

    # def build_index(self, force_rebuild=False):
    #     """Build index of all images in Google Drive folder"""
    #     if self.is_indexed and not force_rebuild:
    #         return

    #     # Try to load cached index
    #     cache_file = "gdrive_index_cache.pkl"
    #     if os.path.exists(cache_file) and not force_rebuild:
    #         with open(cache_file, 'rb') as f:
    #             cached_data = pickle.load(f)
    #             self.image_features = cached_data['features']
    #             self.image_metadata = cached_data['metadata']
    #         self.is_indexed = True
    #         return

    #     # Build new index
    #     self.image_features = {}
    #     self.image_metadata = {}
        
    #     files = self.list_images_in_folder()
        
    #     progress_bar = st.progress(0)
    #     status_text = st.empty()
        
    #     for idx, file in enumerate(files):
    #         try:
    #             status_text.text(f"Processing {file['name']}... ({idx+1}/{len(files)})")
                
    #             # Download and process image
    #             image_buffer = self.download_image(file['id'])
    #             processed_img = self.preprocess_image_from_buffer(image_buffer)
                
    #             # Store features and metadata
    #             self.image_features[file['id']] = processed_img
    #             self.image_metadata[file['id']] = {
    #                 'name': file['name'],
    #                 'mimeType': file['mimeType']
    #             }
                
    #             progress_bar.progress((idx + 1) / len(files))
    #         except Exception as e:
    #             st.warning(f"Error processing {file['name']}: {e}")

    #     # Cache the index
    #     with open(cache_file, 'wb') as f:
    #         pickle.dump({
    #             'features': self.image_features,
    #             'metadata': self.image_metadata
    #         }, f)

    #     progress_bar.empty()
    #     status_text.empty()
    #     self.is_indexed = True

    def compute_similarity(self, img1, img2):
        """Compute SSIM similarity between two images"""
        return ssim(img1, img2)

    def search(self, query_image, similarity_threshold=0.2, limit=3):
        """
        Search for similar images
        query_image: PIL Image or preprocessed numpy array
        """
        if not self.is_indexed:
            self.build_index()

        # Preprocess query image if it's a PIL Image
        if isinstance(query_image, Image.Image):
            query_img = self.preprocess_pil_image(query_image)
        else:
            query_img = query_image

        results = []
        for file_id, processed_img in self.image_features.items():
            try:
                similarity = self.compute_similarity(query_img, processed_img)
                if similarity >= similarity_threshold:
                    results.append((file_id, similarity))
            except Exception as e:
                print(f"Error comparing with {file_id}: {e}")

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_image_url(self, file_id):
        """Get a viewable URL for the image"""
        return f"https://drive.google.com/uc?export=view&id={file_id}"

    def download_image_as_base64(self, file_id):
        """Download image and convert to base64"""
        image_buffer = self.download_image(file_id)
        return base64.b64encode(image_buffer.read()).decode()


# --- Streamlit App ---

st.set_page_config(page_title="Google Drive Image Search", layout="centered")
st.title("🔍 Google Drive Image Search")

credentials_dict = json.loads(st.secrets["gcp"]["service_account"])
folder_id = '1satMPzcyiIQNADcRevQMZz1DaimAshXS'

# Initialize search engine when credentials and folder ID are provided
if credentials_dict and folder_id:
    try:
        
        # Initialize search engine in session state
        if 'search_engine' not in st.session_state:
            st.session_state.search_engine = GoogleDriveImageSearchEngine(
                credentials_dict, 
                folder_id
            )
        
        # Index management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Build/Rebuild Index"):
                with st.spinner("Indexing images from Google Drive..."):
                    st.session_state.search_engine.build_index()
                st.success("Index built successfully!")
        
        with col2:
            if st.session_state.search_engine.is_indexed:
                st.info(f"📊 {len(st.session_state.search_engine.image_features)} images indexed")
        
        st.markdown("---")
        
        # Image input methods
        st.subheader("Search for Similar Images")
        
        tab1, tab2 = st.tabs(["📋 Paste from Clipboard", "📁 Upload Image"])
        
        with tab1:
            st.info("💡 Tip: Take a screenshot (Win+Shift+S or Cmd+Shift+4), then paste here with Ctrl+V / Cmd+V")
            clipboard_image = st.file_uploader(
                "Paste or upload your image:",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="clipboard"
            )
            
            if clipboard_image:
                pil_image = Image.open(clipboard_image)
                st.image(pil_image, caption="Query Image", use_container_width=True)
                
                if st.button("🔍 Search", key="search_clipboard"):
                    with st.spinner("Searching for similar images..."):
                        results = st.session_state.search_engine.search(
                            pil_image,
                            similarity_threshold=0.2,
                            limit=3
                        )
                    
                    if results:
                        st.success(f"Found {len(results)} similar images!")
                        
                        for idx, (file_id, similarity) in enumerate(results):
                            metadata = st.session_state.search_engine.image_metadata[file_id]
                            
                            st.markdown(f"### Match {idx+1}: {metadata['name']}")
                            st.progress(similarity, text=f"Similarity: {similarity:.2%}")
                            
                            # Get image as base64
                            img_base64 = st.session_state.search_engine.download_image_as_base64(file_id)
                            
                            # Display with Fancybox
                            html_code = f"""
                            <html>
                            <head>
                            <link href="https://cdn.jsdelivr.net/npm/@fancyapps/ui@4.0/dist/fancybox.css" rel="stylesheet" />
                            <script src="https://cdn.jsdelivr.net/npm/@fancyapps/ui@4.0/dist/fancybox.umd.js"></script>
                            <style>
                            .image-container {{
                                text-align: center;
                                margin: 10px auto;
                            }}
                            img {{
                                max-width: 100%;
                                max-height: 400px;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                                cursor: zoom-in;
                                border-radius: 8px;
                            }}
                            </style>
                            </head>
                            <body>
                            <div class="image-container">
                                <a href="data:image/png;base64,{img_base64}" data-fancybox="gallery-{idx}" data-caption="{metadata['name']}">
                                    <img src="data:image/png;base64,{img_base64}" alt="{metadata['name']}" />
                                </a>
                            </div>
                            <script>
                            Fancybox.bind("[data-fancybox]", {{
                                Toolbar: {{
                                    display: [
                                        {{ id: "zoom", position: "right" }},
                                        {{ id: "fullscreen", position: "right" }},
                                        {{ id: "close", position: "right" }},
                                    ],
                                }},
                            }});
                            </script>
                            </body>
                            </html>
                            """
                            components.html(html_code, height=450)
                            
                            # Link to open in Google Drive
                            drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                            st.markdown(f"[📂 Open in Google Drive]({drive_url})")
                            st.markdown("---")
                    else:
                        st.warning("No similar images found. Try adjusting the similarity threshold.")
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Choose an image file:",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="upload"
            )
            
            if uploaded_file:
                pil_image = Image.open(uploaded_file)
                st.image(pil_image, caption="Query Image", use_container_width=True)
                
                if st.button("🔍 Search", key="search_upload"):
                    with st.spinner("Searching for similar images..."):
                        results = st.session_state.search_engine.search(
                            pil_image,
                            similarity_threshold=0.2,
                            limit=3
                        )
                    
                    if results:
                        st.success(f"Found {len(results)} similar images!")
                        
                        for idx, (file_id, similarity) in enumerate(results):
                            metadata = st.session_state.search_engine.image_metadata[file_id]
                            
                            st.markdown(f"### Match {idx+1}: {metadata['name']}")
                            st.progress(similarity, text=f"Similarity: {similarity:.2%}")
                            
                            img_base64 = st.session_state.search_engine.download_image_as_base64(file_id)
                            
                            html_code = f"""
                            <html>
                            <head>
                            <link href="https://cdn.jsdelivr.net/npm/@fancyapps/ui@4.0/dist/fancybox.css" rel="stylesheet" />
                            <script src="https://cdn.jsdelivr.net/npm/@fancyapps/ui@4.0/dist/fancybox.umd.js"></script>
                            <style>
                            .image-container {{
                                text-align: center;
                                margin: 10px auto;
                            }}
                            img {{
                                max-width: 100%;
                                max-height: 400px;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                                cursor: zoom-in;
                                border-radius: 8px;
                            }}
                            </style>
                            </head>
                            <body>
                            <div class="image-container">
                                <a href="data:image/png;base64,{img_base64}" data-fancybox="gallery-{idx}" data-caption="{metadata['name']}">
                                    <img src="data:image/png;base64,{img_base64}" alt="{metadata['name']}" />
                                </a>
                            </div>
                            <script>
                            Fancybox.bind("[data-fancybox]", {{
                                Toolbar: {{
                                    display: [
                                        {{ id: "zoom", position: "right" }},
                                        {{ id: "fullscreen", position: "right" }},
                                        {{ id: "close", position: "right" }},
                                    ],
                                }},
                            }});
                            </script>
                            </body>
                            </html>
                            """
                            components.html(html_code, height=450)
                            
                            drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                            st.markdown(f"[📂 Open in Google Drive]({drive_url})")
                            st.markdown("---")
                    else:
                        st.warning("No similar images found.")
    except Exception as e:
        st.error(f"Error initializing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("👈 Please configure your Google Drive credentials and folder ID in the sidebar to get started.")
    
    with st.expander("📖 Setup Instructions"):
        st.markdown("""
        ### How to set up:
        
        1. **Create a Google Cloud Project:**
           - Go to [Google Cloud Console](https://console.cloud.google.com/)
           - Create a new project or select an existing one
        
        2. **Enable Google Drive API:**
           - In the API Library, search for "Google Drive API"
           - Click "Enable"
        
        3. **Create Service Account:**
           - Go to "IAM & Admin" > "Service Accounts"
           - Click "Create Service Account"
           - Download the JSON key file
        
        4. **Share your Google Drive folder:**
           - Right-click your folder > Share
           - Add the service account email (from the JSON)
           - Give it "Viewer" permission
        
        5. **Get Folder ID:**
           - Open your folder in Google Drive
           - Copy the ID from the URL
        
        6. **Paste credentials and folder ID** in the sidebar
        """)
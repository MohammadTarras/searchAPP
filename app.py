import streamlit as st
import os
import glob
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import pickle
import time
import numpy as np
import base64
import streamlit.components.v1 as components

class ImageSearchEngine:
    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.image_features = {}
        self.is_indexed = False

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))
        return resized

    def build_index(self, force_rebuild=False):
        if self.is_indexed and not force_rebuild:
            return

        index_file = os.path.join(self.image_directory, "image_index.pkl")

        if os.path.exists(index_file) and not force_rebuild:
            with open(index_file, 'rb') as f:
                self.image_features = pickle.load(f)
            self.is_indexed = True
            return

        self.image_features = {}
        for filename in os.listdir(self.image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                image_path = os.path.join(self.image_directory, filename)
                try:
                    processed_img = self.preprocess_image(image_path)
                    self.image_features[image_path] = processed_img
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        with open(index_file, 'wb') as f:
            pickle.dump(self.image_features, f)

        self.is_indexed = True

    def compute_similarity(self, img1, img2):
        return ssim(img1, img2)

    def search(self, query_image_path, similarity_threshold=0.2, limit=3):
        if not self.is_indexed:
            self.build_index()

        query_img = self.preprocess_image(query_image_path)

        results = []
        for image_path, processed_img in self.image_features.items():
            try:
                similarity = self.compute_similarity(query_img, processed_img)
                if similarity >= similarity_threshold:
                    results.append((image_path, similarity))
            except Exception as e:
                print(f"Error comparing with {image_path}: {e}")

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

def get_most_recent_image(directory):
    image_files = glob.glob(os.path.join(directory, "*.[jpJP][pnPN]*"))
    if not image_files:
        return None
    return max(image_files, key=os.path.getmtime)

# --- Streamlit App ---

# Configuration
st.set_page_config(page_title="Image Search App", layout="centered")
st.title("🔍 Screenshot Similarity Finder")

# Directories
screenshot_directory = "C:/Users/User/Pictures/Screenshots"
image_directory = "G:/My Drive/Processed Cheese Images (Arrow)/Uploaded"

# Initialize Search Engine
search_engine = ImageSearchEngine(image_directory)

# Button to regenerate index
if st.button("🔄 Regenerate Index"):
    with st.spinner("Re-indexing images..."):
        search_engine.build_index(force_rebuild=True)
    st.success("Index regenerated successfully!")

# Button to get most similar image
if st.button("📷 Get Image"):
    latest_image = get_most_recent_image(screenshot_directory)
    if latest_image:
        with st.spinner("Searching for similar image..."):
            search_engine.build_index()
            results = search_engine.search(latest_image)

        if results:
            print(results)
            top_match, similarity = results[0]
            with open(top_match, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode()
                
                # Only show Fancybox fullscreen view
                html_code = f"""
                <html>
                <head>
                <link href="https://cdn.jsdelivr.net/npm/@fancyapps/ui@4.0/dist/fancybox.css" rel="stylesheet" />
                <script src="https://cdn.jsdelivr.net/npm/@fancyapps/ui@4.0/dist/fancybox.umd.js"></script>
                <style>
                .image-container {{
                    text-align: center;
                    margin: 20px auto;
                }}
                img {{
                    max-width: 100%;
                    max-height: 600px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    cursor: zoom-in;
                }}
                </style>
                </head>
                <body>
                <div class="image-container">
                    <a href="data:image/png;base64,{encoded}" data-fancybox="gallery" data-caption="Similar Image">
                        <img src="data:image/png;base64,{encoded}" alt="Similar image" />
                    </a>
                </div>

                <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    // Initialize Fancybox
                    Fancybox.bind("[data-fancybox]", {{
                        // Custom options for Fancybox
                        Toolbar: {{
                            display: [
                                {{ id: "prev", position: "center" }},
                                {{ id: "counter", position: "center" }},
                                {{ id: "next", position: "center" }},
                                {{ id: "zoom", position: "right" }},
                                {{ id: "fullscreen", position: "right" }},
                                {{ id: "close", position: "right" }},
                            ],
                        }},
                        Images: {{
                            zoom: true,
                        }},
                    }});
                    
                    // Auto-open Fancybox (if you want the image to open automatically)
                    // Uncomment the lines below if you want this behavior
                    /*
                    Fancybox.show([{{
                        src: "data:image/png;base64,{encoded}",
                        type: "image",
                        caption: "Similar Image"
                    }}]);
                    */
                }});
                </script>
                </body>
                </html>
                """
                # Render HTML with full JS support
                components.html(html_code, height=650)
                top_match = top_match.split("/")[-1].split("\\")[-1]
                st.info(f"Most similar image is: {top_match}")
        else:
            st.warning("No similar images found.")
    else:
        st.error("No images found in screenshot directory.")
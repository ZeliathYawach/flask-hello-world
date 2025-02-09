from flask import Flask, request, jsonify
from gradio_client import Client, file  # Note: file() is deprecated; consider using handle_file() as per the warning.
import os
import httpx

app = Flask(__name__)

# Replace with your Imgbb API key
IMGBB_API_KEY = "df5461c520f518a4417bd57a8446453b"  # Replace with your actual Imgbb API key

def upload_to_imgbb(file_path):
    """Uploads a file to Imgbb and returns the public URL."""
    url = "https://api.imgbb.com/1/upload"
    with open(file_path, "rb") as file_obj:
        files = {"image": file_obj}
        data = {"key": IMGBB_API_KEY}
        response = httpx.post(url, data=data, files=files)
        response_json = response.json()

        if response.status_code == 200 and response_json.get("success"):
            return response_json["data"]["url"]
        else:
            raise Exception(f"Failed to upload to Imgbb: {response_json.get('error', 'Unknown error')}")

@app.route('/process-image', methods=['POST'])
def process_image():
    processed_image_paths = None  # Will hold a list of one or more image paths for cleanup
    try:
        # Expecting JSON input with 'src_image_url' and 'ref_image_url'
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input, expecting JSON payload"}), 400
        if 'src_image_url' not in data or 'ref_image_url' not in data:
            return jsonify({"error": "Both 'src_image_url' and 'ref_image_url' are required"}), 400

        src_image_url = data['src_image_url']
        ref_image_url = data['ref_image_url']

        # Initialize the Gradio client for the model
        client = Client("franciszzj/Leffa")
        result = client.predict(
            src_image_path=file(src_image_url),  # Using the URL directly
            ref_image_path=file(ref_image_url),  # Using the URL directly
            ref_acceleration=False,   # Boolean value
            step=30,
            scale=2.5,
            seed=42,
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",
            vt_repaint=False,         # Boolean value
            api_name="/leffa_predict_vt"
        )

        # Extract the processed image paths from the result.
        # The result may be a list/tuple of image paths or a dict with a key "image_path"
        if isinstance(result, (list, tuple)):
            processed_image_paths = list(result)
        elif isinstance(result, dict):
            image_path = result.get("image_path")
            if isinstance(image_path, list):
                processed_image_paths = image_path
            else:
                processed_image_paths = [image_path]
        else:
            return jsonify({"error": "Unexpected result format from prediction"}), 500

        # Ensure we have at least one valid image path
        if not processed_image_paths or not processed_image_paths[0]:
            return jsonify({"error": "Processed image path not found in response"}), 500

        # Use the first image for uploading
        imgbb_url = upload_to_imgbb(processed_image_paths[0])

        return jsonify({"processed_image_url": imgbb_url})

    except httpx.ProxyError as e:
        print(f"Proxy error occurred: {e}")
        return jsonify({"error": "Proxy error occurred"}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred", "details": str(e)}), 500
    finally:
        # Cleanup all processed image files
        if processed_image_paths:
            for path in processed_image_paths:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Removed temporary file: {path}")
                    except Exception as cleanup_error:
                        print(f"Error removing file {path}: {cleanup_error}")

if __name__ == '__main__':
    app.run(debug=True)

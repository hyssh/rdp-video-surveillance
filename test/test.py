"""
This script tests the Florence API by sending a base64 encoded image
and receiving parsed content, specifically looking for unique icons in the response.
"""
import base64
import requests
import time


def test_run(num_of_testing: int = 10):
    """
    With NVIDIA GeForece RTX 4050 Laptop GPU, this might take around 22 seconds
    """
    image_path = "images\\demo_video_582025_0000000000.jpg"
    florence_url = "http://localhost:8000/parse/"

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        for i in range(num_of_testing):
            print(f"Testing {i + 1}/{num_of_testing}...")
            # Send the base64 image to the Florence API
            response = requests.post(florence_url, json={"base64_image": base64_image})
            if response.status_code == 200:
                icon_sets = set()
                json_response = response.json()
                del json_response['som_image_base64']
                for item in json_response['parsed_content_list']:
                    if 'bbox' in item:
                        del item['bbox']
                
                # print(json_response)
                for item in json_response['parsed_content_list']:
                    if item['type'] == 'icon':
                        icon_sets.add(item['content'])

                print(f"Number of unique icons: {len(icon_sets)}")
                print(f"Unique icons: {icon_sets}")        
            else:
                print(f"Error: {response.status_code}")


if __name__ == "__main__":
    start_time = time.time()
    test_run()
    end_time = time.time()
    print("="*33)
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("="*33)
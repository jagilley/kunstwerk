import requests
import json

def generate_anime_image(prompt: str, seed: int = 346593, candidates: int = 4) -> requests.Response:
    """
    Generate anime-style images using Google's ImageFX API.
    
    Args:
        prompt: The text description of the image to generate
        seed: Random seed for reproducibility
        candidates: Number of images to generate
        
    Returns:
        Response from the API
    """
    url = 'https://aisandbox-pa.googleapis.com/v1:runImageFx'
    
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.5',
        'authorization': 'Bearer ya29.a0AXeO80TU_XrzpnmHRkrNW6BeYRat0U0sKX2BGQzWycRMTWTrJqE_TNeFnfrrID-UYQQV_Di41YV0grCYcLsIhAoHsmox1rhfLqUffpnceExQS_yGxcx1IPjYIn9f4bxWRE18TRFXu1CN7pdefQvQif6lAJdMGp4k5cXNLMnIMXjB3cBfg1DGoH8JQ_nXjh8gh5858g5Q5HyfvnZ0bHf7L7joAi0TGO8s6f50e5mIgefY6VMo0a9v5S612i1vABD2uJQu4i1Jl-8vfBTNw4cvAKTVCMX1ScfOZ7DNrP5UCdl9XwnL99lpP8ilNE8rp4HQgzFongnEd2pAlcpCO6UUptzqNppSLIC2RDHIqHTZUi7oOjW4E04ZaIzRU9ZWw4AGo0dKTqcFSPddJA8NTYNr8RDB3cnJnrse3LwaRlif9gaCgYKAc8SARASFQHGX2MiILgjM7vPf5PXRL5dpGeHTA0433',
        'content-type': 'text/plain;charset=UTF-8',
        'origin': 'https://labs.google',
        'priority': 'u=1, i',
        'referer': 'https://labs.google/',
        'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Brave";v="132"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
    }
    
    data = {
        "userInput": {
            "candidatesCount": candidates,
            "prompts": [prompt],
            "seed": seed
        },
        "clientContext": {
            "sessionId": ";1737827470214",
            "tool": "IMAGE_FX"
        },
        "modelInput": {
            "modelNameType": "IMAGEN_3_1"
        },
        "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response

if __name__ == "__main__":
    # Example usage
    prompt = "Valhalla goes up in flames as the gods look on. Still from Götterdämmerung, cinematic artistic anime"
    response = generate_anime_image(prompt)
    print(f"Status code: {response.status_code}")
    print(response.json())

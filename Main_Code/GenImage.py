def t2img(input2):
    import requests
    
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": "Bearer API_KEY"}
    
    def query(payload):
    	response = requests.post(API_URL, headers=headers, json=payload)
    	return response.content
    image_bytes = query({
    	"inputs": input2,
    })
    # You can access the image with PIL.Image for example
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))
    image.show()
a=input()
t2img(a)

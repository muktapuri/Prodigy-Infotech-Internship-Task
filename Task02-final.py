#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from diffusers import StableDiffusionPipeline


# In[2]:


# Step 1: Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipeline.to(device)


# In[3]:


# Step 2: Define your text prompt and generate an image
prompt = "A futuristic city skyline at sunset"
try:
    image = pipeline(prompt, guidance_scale=7.5).images[0]
    image.save("generated_image.png")
    image.show()
except Exception as e:
    print(f"An error occurred during generation: {e}")


# In[6]:


try:
    image = pipeline(prompt, guidance_scale=8.5, num_inference_steps=100).images[0]
    image.save("refined_generated_image.png")
    image.show()
except Exception as e:
    print(f"An error occurred during fine-tuning: {e}")


# In[ ]:


#  Generate images from multiple prompts
prompts = ["A gardern full of flower","fairy she's helping a child ","An thre is cloud on earth "kids playing football"]
for i, prompt in enumerate(prompts):
    try:
        image = pipeline(prompt, guidance_scale=7.5).images[0]
        image.save(f"generated_image_{i}.png")
        image.show()
    except Exception as e:
        print(f"An error occurred during generation for prompt {i}: {e}")


# In[ ]:


#  Generate images from multiple prompts
prompts = ["A serene mountain landscape", "A dragon flying over a castle", "An astronaut riding a horse in space"]
for i, prompt in enumerate(prompts):
    try:
        image = pipeline(prompt, guidance_scale=7.5).images[0]
        image.save(f"generated_image_{i}.png")
        image.show()
    except Exception as e:
        print(f"An error occurred during generation for prompt {i}: {e}")


# In[ ]:





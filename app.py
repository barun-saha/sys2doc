import logging
import os

import PIL
import requests
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
)

SUPPORTED_FILE_EXTENSIONS = ['png', 'jpg', 'jpeg']
IMAGE_PROMPT = (
    # 'You are a systems expert.'
    ' The provided image relates to a system.'
    # ' Refuse to answer if the provided image is not related to any system or software in any way.'
    ' The system\'s image could be of any type, such as architecture diagram, flowchart, state machine, and so on.'
    ' Based SOLELY on the image, describe the system and its different components in detail.'
    ' You should not use any prior knowledge except for universal truths.'
    ' If relevant, describe how the relevant components interact and how information flows.'
    ' In case the image contains or relates to anything inappropriate'
    ' including, but not limited to, violence, hatred, malice, and criminality,'
    ' DO NOT generate an answer and simply say that you are not allowed to describe.'
)

GENERATION_CONFIG = {
    "temperature": 0,
    "top_p": 0.5,
    "top_k": 10,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]


@st.cache_resource
def get_gemini_model():
    """
    Get the Gemini model.

    :return: The model
    """

    return genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS
    )


def get_image_description(image: PIL.Image) -> str:
    """
    Use Gemini Pro Vision LMM to generate a response.

    :param image: The image to use
    :return: The description based on the image
    """

    model = get_gemini_model()
    response = model.generate_content([IMAGE_PROMPT, image], stream=False).text
    # print(f'> {response=}')

    return response


# The page
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

st.title('Sys2Doc')
st.header(
    'Generate Documentation Based on System Diagram — powered by Gemini 1.5 Flash'
)

uploaded_file = st.file_uploader(
    'Choose an image file (PNG, JPG, or JPEG) that depicts your system,'
    ' for example, architecture, state machine, flow diagram, and so on',
    type=SUPPORTED_FILE_EXTENSIONS
)

if uploaded_file is not None:
    # Show the uploaded image & related info
    the_img = None
    file_details = None

    try:
        the_img = PIL.Image.open(uploaded_file)
        file_details = {
            'file_name': uploaded_file.name,
            'file_type': uploaded_file.type,
            'file_size': uploaded_file.size
        }

        if the_img and the_img.mode in ('RGBA', 'P'):
            the_img = the_img.convert('RGB')

        st.header('Image')

        if file_details:
            st.write(file_details)

        st.image(the_img, width=250)
        description = get_image_description(the_img)
        st.header('Description')
        st.write(description)
        logging.debug(description)
        logging.info('Done!')
    except PIL.UnidentifiedImageError as uie:
        st.error(f'An error occurred while loading the image: {uie}')
        logging.debug(f'An error occurred while loading the image: {uie}\n'
                      f'File details: {file_details}')
    except requests.exceptions.MissingSchema as ms:
        st.error(f'Please specify a proper URL for the image.')
    finally:
        st.divider()
        st.write('Sys2Doc is an experimental prototype, with no guarantee provided whatsoever.'
                 ' Use it fairly, responsibly, and with care.')

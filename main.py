import streamlit as st
import pickle
import re
import nltk
import base64
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open(r'C:\Users\Administrator\Downloads\clf.pkl', 'rb'))
tfidfd = pickle.load(open(r'C:\Users\Administrator\Downloads\tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Function to extract video ID from YouTube video link
def extract_video_id(youtube_link):
    query = urlparse(youtube_link)
    if query.hostname == 'www.youtube.com':
        if 'v' in query.query:
            return parse_qs(query.query)['v'][0]
    elif query.hostname == 'youtu.be':
        return query.path[1:]
    return None

# Web app
def main():
    st.title("Resume Screener")

    # Replace the link with your YouTube video link
    youtube_link = "https://www.youtube.com/watch?v=Tt08KmFfIYQ"

    # Extract video ID from the YouTube link
    general_tips_video_id = extract_video_id(youtube_link)

    if general_tips_video_id is None:
        st.warning("Invalid YouTube link. Please provide a valid YouTube video link.")
        return

    # Create a sidebar for upload
    st.sidebar.header("Upload Resume")
    uploaded_file = st.sidebar.file_uploader('Choose a resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        # Highlight the predicted category label
        highlighted_category = f'<span style="font-size: 20px; color: #FF5733;">{category_name}</span>'
        st.markdown(f"Predicted Category: {highlighted_category}", unsafe_allow_html=True)

        # Increase font size and style for the message
        message_html = '<p style="font-size: 18px; font-weight: bold;">Click on the below video for some exciting resume building tips!!</p>'
        st.markdown(message_html, unsafe_allow_html=True)

        # Display YouTube video thumbnail for general tips with clickable link
        youtube_thumbnail_url = f"https://img.youtube.com/vi/{general_tips_video_id}/hqdefault.jpg"
        youtube_thumbnail_html = f'<a href="{youtube_link}" target="_blank"><img src="{youtube_thumbnail_url}" alt="YouTube Thumbnail" width="700" height="450"></a>'
        st.markdown(youtube_thumbnail_html, unsafe_allow_html=True)

# Python main
if __name__ == "__main__":
    main()


import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
def classify_news(article):
    # Tokenize and encode the article
    inputs = tokenizer.encode_plus(
        article,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label
    predicted_label = torch.argmax(outputs.logits).item()
    labels = ['real', 'fake']
    predicted_class = labels[predicted_label]

    return predicted_class
def main():
    # Set the page title and background color
    st.set_page_config(page_title='MisinfoWatch Classifier', page_icon='📰', layout='wide', initial_sidebar_state='auto')

    # Set the background color
    page_bg = '''
    <style>
    body {
        background-color: lightblue;
    }
    </style>
    '''
    st.markdown(page_bg, unsafe_allow_html=True)

    # Add a title and description
    st.title('MisInformation Classifier')
    st.write('Enter the text of the news article to classify if it is real or fake.')

    # Get user input
    news_article = st.text_area('News Article')

    # Classify the news article when a button is clicked
    if st.button('Classify'):
        predicted_class = classify_news(news_article)
        st.write('Predicted class:', predicted_class)

if __name__ == '__main__':
    main()

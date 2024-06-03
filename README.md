
# Language Learning App Prototype

This project represents an initial prototype for a language learning application. The app's core functionality is to predict and translate words that users might find challenging, based on their proficiency level and the words they mark as unknown. Utilizing linguistic features such as word frequency, morphological similarity, and part of speech, this prototype demonstrates the feasibility of personalized language learning through heuristic methods.

## Purpose

The primary goal of this prototype is to serve as a proof of concept for the language learning app. By leveraging heuristic methods, it aims to facilitate language learning through personalized content, helping users to better understand and engage with texts in a foreign language. This approach not only predicts unknown words but also provides translations and contextual sentences to aid comprehension.

## Features

The prototype offers several key features:
- Predicts unknown words based on the user's proficiency level and their input of marked unknown words.
- Utilizes various linguistic features, including word frequency, morphological similarity, and part of speech, to make predictions.
- Translates these unknown words and provides contextual sentences where they appear.
- Highlights the unknown words and their translations within the text, making it easier for users to learn and understand new vocabulary.

## Dependencies and Installation

To run this prototype, several dependencies need to be installed, including `gradio`, `stanza`, `deep-translator`, `nltk`, `wordfreq`, `rapidfuzz`, and `pandas`. You can install all required dependencies using the following pip command:

```sh
pip install gradio stanza deep-translator nltk wordfreq rapidfuzz pandas
```

The app can be deployed in [Hugging Face Spaces](https://huggingface.co/spaces/AiManatee/language_learning_app_prototype) or run locally using a Google Colab notebook with a Gradio interface.

## Usage

Users can interact with the app in two main ways: through Hugging Face Spaces or by running a Google Colab notebook. For Hugging Face Spaces, simply access the app via the provided link. To run the app in Google Colab, open the notebook containing this script, run all cells to initialize the app, and use the Gradio interface to interact with it.

The input to the app consists of text in a foreign language (currently only Spanish), and the output is the text presented paragraph by paragraph with unknown words predicted and translated.

## Notes

While this fast initial prototype demonstrates the potential of a heuristic-based model for language learning, it has some efficiency limitations due to the complexity of NLP tasks. Additionally, the quality of contextual translations could be better and will significantly improve with the implementation of a more robust API for translations. This prototype currently supports only Spanish and serves as a preliminary step toward a more comprehensive application. Future iterations, incorporating machine learning models, will greatly enhance the accuracy and efficiency of predicting unknown words and providing precise translations.

## Future Steps

The next steps for this project will involve creating a machine learning classifier to improve the efficiency and quality of predictions. By incorporating machine learning techniques, we aim to enhance the app's ability to accurately predict unknown words and provide more precise translations.

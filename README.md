This project is an llm application that is used to search, summarise, translate, site research papers
It fetches the papaers from arxiv website and uses google flan t5 large model to summarise and translate.
The application has 2 options, to search the paper online or upload a paper 
A chatbot is provided where you can select the respective paper and ask the bot questions regarding the paper and it will answer
The ui is made using stramlit library
The google/flan-t5-large model is downloaded to the local device and is used, for the first time the model is downloaded if not available and if available, runs without downloading

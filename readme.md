# My Profile Chatbot

A chatbot about my profile, experience, education and skills

## Project Structure

- basic.py: 
    - main workflow of the chatbot in cli
    - run with `python qa_chain_cli.py`
- ui_qa.py
    - streamlit app for QA chatbot
    - run with `streamlit run streamlit_app.py`


## Todos:

- update readme for project structure, esp data, venv and env
- add more data
    - add updated cv with project
    - educational docs
    - experience letters
    - project docs/ reports/ readme
    - linkedin stuff: get more from linkedin csv
- better llm selection: using gemini for now
    - add ui selection for gemini free models
    - improve qa pipeline and prompt
    - add prompt history for agent mem
- make router chains for better response
    - add router chains for education, skills, experience and default
- UI: improve UI
    - chat sequence
- deploy:
    - improve the huggingface guthub repo hosting setup
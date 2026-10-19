import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datasets import load_dataset

def normalize_title( title : str ) -> str: 
    return title.replace("\xa0", " ")

# Importig datasets
ds = load_dataset("AdamLucek/youtube-titles")

# Filtering data
dataset = ds["train"].shuffle(seed = 67).select(range(1020))
titles = dataset["video_title"]

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/forms.body","https://www.googleapis.com/auth/drive.file"]

creds = None

if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file(
        "token.json",
        SCOPES
    )

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json",
            SCOPES
        )

        creds = flow.run_local_server(port=0)

    with open("token.json", "w") as token:
        token.write(creds.to_json())


form_service = build(
    "forms",
    "v1",
    credentials=creds
)

drive_service = build(
    "drive",
    "v3",
    credentials=creds
)

results = drive_service.files().list(orderBy = "name_natural", q = "name contains 'Productivity-Enhancer classification form'").execute()

labels = []
total_responses = 0
current_response = 0
input_to_label_mapping = {}

for i in range(34): #check the 34 forms

    #get response api endpoint
    form_response = form_service.forms().responses().list(formId = results["files"][i]["id"]).execute()

    #get form structure api endpoint
    form_id = results["files"][i]["id"]
    form = form_service.forms().get(formId=form_id).execute()

    #mapping questions to question_ids
    question_mapping = {}

    for item in form["items"]:
        if "questionItem" in item:
            question_id = item["questionItem"]["question"]["questionId"]
            question_title = item["title"]

            question_mapping[question_id] = question_title

    responses = form_response["responses"]

    for question in responses[0]["answers"]: #check the 30 questions

        response = responses[0] #take the first response

        answer = response["answers"].get(question)

        choice = answer["textAnswers"]["answers"][0]["value"]

        if choice == "Productive":
            current_response = 0
        elif choice == "Educational":
            current_response = 1
        else:
            current_response = 2

        input_to_label_mapping[normalize_title(question_mapping[question])] = current_response

        labels.append(current_response)

    total_responses += len(responses)

label_col = []

for i in range(1020):

    title = normalize_title(titles[i])

    label_col.append(input_to_label_mapping[title])

dataset = dataset.add_column("productivity_label", label_col)
dataset = dataset.select_columns(["video_title", "video_description", "productivity_label"])
split_dataset = dataset.train_test_split(0.15, seed = 42 , shuffle = False)

split_dataset.push_to_hub("Rodrigo-Martin/productivity_enhancer")

import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datasets import load_dataset

# Importig datasets
ds = load_dataset("AdamLucek/youtube-titles")

# Filtering data
dataset = ds["train"].shuffle(seed = 67).select(range(1020))
dataset = dataset[:]["video_title"]

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

start = 0
end = 30

description = """Please classify each YouTube video title according to how productive
watching the video would be.

Productive:
The video is at least educational about a scientific topic (Computer Science, Mathematics, Engineering, ...) 
and may help develop a useful skill in any of these fields.

Educational:
The video is educational i.e you learn something usefull by watching it

Unproductive:
The video is primarily entertainment or distraction and provides little
educational or practical value.

Please judge based only on the title shown.
"""

for i in range(34):
    form_info = {
        "info": {
            "title": f"Productivity-Enhancer classification form [{i+1}]"
        }
    }

    form = form_service.forms().create(body=form_info).execute()
    form_id = form["formId"]

    drive_service.files().update(
        fileId=form_id,
        body={
            "name": f"Productivity-Enhancer classification form [{i + 1}]"
        }
    ).execute()

    examples = dataset[start : end]

    form_service.forms().batchUpdate(
        formId=form_id,
        body={
            "requests": [
                {
                    "updateFormInfo": {
                        "info": {
                            "description": description
                        },
                        "updateMask": "description"
                    }
                }
            ]
        }
    ).execute()

    requests = []

    for j in range(30):
       requests.append({
            "createItem": {
                "item": {
                    "title": examples[j],
                    "questionItem": {
                        "question": {
                            "required": True,
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": [
                                    {"value": "Productive"},
                                    {"value": "Educational"},
                                    {"value": "Unproductive"}
                                ],
                                "shuffle": False
                            }
                        }
                    }
                },
                "location": {
                    "index": j
                }
            }
        })

    form_service.forms().batchUpdate(
        formId = form_id,
        body = {"requests": requests}
    ).execute()

    start += 30
    end += 30

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'backend'
src_path = f"{project_name}/src"

list_of_files = [
    f"{project_name}/.env", f"{project_name}/.env.example",
    f"{project_name}/docker-compose.yml", f"{project_name}/Dockerfile",
    f"{project_name}/requirements.txt", f"{project_name}/setup.py",
    f"{project_name}/template.py",

    f"{src_path}/app/__init__.py",
    f"{src_path}/app/main.py",

    f"{src_path}/app/api/__init__.py",
    f"{src_path}/app/api/dependencies.py",
    f"{src_path}/app/api/v1/__init__.py",
    f"{src_path}/app/api/v1/endpoints/__init__.py",
    f"{src_path}/app/api/v1/endpoints/auth.py",
    f"{src_path}/app/api/v1/endpoints/users.py",
    f"{src_path}/app/api/v1/endpoints/leads.py",
    f"{src_path}/app/api/v1/endpoints/content.py",
    f"{src_path}/app/api/v1/endpoints/chatbot.py",
    f"{src_path}/app/api/v1/endpoints/social.py",
    f"{src_path}/app/api/v1/router.py",

    f"{src_path}/app/core/__init__.py",
    f"{src_path}/app/core/config.py",
    f"{src_path}/app/core/security.py",
    f"{src_path}/app/core/exceptions.py",

    f"{src_path}/app/db/__init__.py",
    f"{src_path}/app/db/session.py",
    f"{src_path}/app/db/base_class.py",

    f"{src_path}/app/models/__init__.py",
    f"{src_path}/app/models/user.py",
    f"{src_path}/app/models/lead.py",
    f"{src_path}/app/models/lead_score.py",
    f"{src_path}/app/models/content.py",
    f"{src_path}/app/models/chatbot.py",
    f"{src_path}/app/models/social_media.py",

    f"{src_path}/app/schemas/__init__.py",
    f"{src_path}/app/schemas/user.py",
    f"{src_path}/app/schemas/lead.py",
    f"{src_path}/app/schemas/content.py",
    f"{src_path}/app/schemas/chatbot.py",
    f"{src_path}/app/schemas/social.py",
    f"{src_path}/app/schemas/token.py",

    f"{src_path}/app/crud/__init__.py",
    f"{src_path}/app/crud/base.py",
    f"{src_path}/app/crud/user.py",
    f"{src_path}/app/crud/lead.py",
    f"{src_path}/app/crud/content.py",
    f"{src_path}/app/crud/chatbot.py",
    f"{src_path}/app/crud/social.py",

    f"{src_path}/app/services/__init__.py",
    f"{src_path}/app/services/lead_scoring.py",
    f"{src_path}/app/services/content_generation.py",
    f"{src_path}/app/services/chatbot_service.py",
    f"{src_path}/app/services/social_media.py",

    f"{src_path}/app/utils/__init__.py",
    f"{src_path}/app/utils/logging.py",

    # f"{src_path}/alembic/versions/.gitkeep",
    # f"{src_path}/alembic/env.py",
    # f"{src_path}/alembic/alembic.ini",

    f"{src_path}/tests/__init__.py",
    f"{src_path}/tests/conftest.py",
    f"{src_path}/tests/test_api/__init__.py",
    f"{src_path}/tests/test_api/test_auth.py",
    f"{src_path}/tests/test_api/test_users.py",
    f"{src_path}/tests/test_api/test_leads.py",
    f"{src_path}/tests/test_api/test_content.py",
    f"{src_path}/tests/test_api/test_chatbot.py",
    f"{src_path}/tests/test_api/test_social.py",
    f"{src_path}/tests/test_services/__init__.py",
    f"{src_path}/tests/test_services/test_lead_scoring.py",
    f"{src_path}/tests/test_services/test_content_generation.py",
    f"{src_path}/tests/test_services/test_chatbot.py",
    f"{src_path}/tests/test_services/test_social_media.py",

    f"{src_path}/scripts/start.sh",
    f"{src_path}/scripts/seed_db.py",

    f"{src_path}/docs/api.md",
    f"{src_path}/docs/development.md",
]

for file in list_of_files:
    file = Path(file)
    filedir, filename = os.path.split(file)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(file)) or (os.path.getsize(file) == 0):
        with open(file, 'w') as f:
            pass
        logging.info(f"Creating empty file: {file}")
    else:
        logging.info(f"File already exists: {file}")

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from auth_ui import configure_page, persist_auth_state, render_shell
from database import authenticate_user, init_db


configure_page("TALEXA Login")
init_db()

auth_message: str | None = None
auth_message_type = "info"

form_container = render_shell("Welcome to Talexa! Your personal AI lecture")

with form_container:
    email = st.text_input("Email", placeholder="Email address", label_visibility="collapsed")
    password = st.text_input("Password", placeholder="Password", type="password", label_visibility="collapsed")

    if st.button("log in", use_container_width=True):
        clean_email = email.strip()
        clean_password = password.strip()

        if not clean_email or not clean_password:
            auth_message = "Please enter both email and password."
            auth_message_type = "error"
        else:
            is_valid, message, user = authenticate_user(clean_email, clean_password)
            if is_valid:
                persist_auth_state(user["user_ID"], user["Email"])
                st.switch_page("pages/terms_and_conditions.py")
            else:
                auth_message = message
                auth_message_type = "error"

    if auth_message:
        if auth_message_type == "success":
            st.success(auth_message)
        else:
            st.error(auth_message)

    st.markdown('<div class="divider"><span>or</span></div>', unsafe_allow_html=True)

    if st.button("sign up", use_container_width=True):
        st.switch_page("pages/sign_up.py")

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from auth_ui import configure_page, render_shell
from database import create_user, get_user_by_email, init_db


configure_page("TALEXA Sign Up")
init_db()

signup_message: str | None = None
signup_message_type = "info"

form_container = render_shell("Create account:")

with form_container:
    email = st.text_input("Email", placeholder="Email", label_visibility="collapsed")
    password = st.text_input("Password", placeholder="Password", type="password", label_visibility="collapsed")

    if st.button("create account", use_container_width=True):
        clean_email = email.strip()
        clean_password = password.strip()

        if not clean_email or not clean_password:
            signup_message = "Please enter both email and password."
            signup_message_type = "error"
        elif get_user_by_email(clean_email):
            signup_message = "An account already exists for this email."
            signup_message_type = "error"
        elif create_user(clean_email, clean_password):
            signup_message = "Account created successfully. You can now log in."
            signup_message_type = "success"
        else:
            signup_message = "Unable to create the account. Please try again."
            signup_message_type = "error"

    if signup_message:
        if signup_message_type == "success":
            st.success(signup_message)
        else:
            st.error(signup_message)

    st.markdown('<div class="divider"><span>or</span></div>', unsafe_allow_html=True)

    if st.button("back to log in", use_container_width=True):
        st.switch_page("pages/login.py")

from __future__ import annotations

import streamlit as st

from auth_ui import configure_page
from database import init_db


configure_page("TALEXA")
init_db()

st.switch_page("pages/login.py")

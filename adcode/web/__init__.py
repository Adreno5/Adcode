"""Serve the full-screen TUI to browsers on the local network.

The terminal UI is not re-implemented for the web: the same AdCodeApp runs
inside the server, and only its two terminal edges are swapped out.
WebScreen publishes the composited cell grid instead of ANSI to a console, and
WebInput turns browser keystrokes into the same KeyEvent objects the console
reader produces.
"""

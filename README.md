# AdCode

AdCode is a configurable, full-screen terminal coding agent. It streams from
OpenAI-compatible and Anthropic model APIs, runs a ReAct loop with tools
(files, search, a persistent shell, HTTP, git diffs), delegates work to
sub-agents and teams, loads MCP servers, and reads skills from
`~/.agents/skills`.

The interface is a small software renderer: every widget draws into a shared
cell canvas with an alpha channel, the compositor blends the frame, and the
whole screen is written to the terminal in one pass.

## Highlights

- **Full-screen canvas UI** -- a unified style built from white and light grey
  layers, with the accent colour used only for small markers (`>`, `│`, `›`).
  Every frame is composited in software and printed at once.
- **Alpha-aware rendering bus** -- widgets call `draw_text`, `fill_rect`,
  `box`, and friends; colours blend through 8-bit alpha, which drives the
  fading thinking window, dimmed modal backdrops, and gradients.
- **App-managed scrolling** -- the transcript is a virtual scrollback with a
  scrollbar, mouse-wheel support, PageUp/PageDown, and follow-tail behaviour.
- **Streaming ReAct loop** -- thinking, answers, tool calls, and usage stream
  from worker threads; the UI stays responsive and Esc interrupts a turn or a
  running shell command.
- **Automatic context compaction** -- old complete turns are summarized
  through a streaming model call, archived locally, and replaced atomically.
- **Provider and model management** -- `/provider add` fetches the model list,
  offers a filterable multi-select table, accepts custom model ids, and fills
  capabilities (vision, context window, max output tokens, reasoning efforts)
  from [models.dev](https://models.dev). Every model can be edited by hand.
- **Sub-agents and agent teams** -- configurable agent types, background ReAct
  workers, a shared workspace, mailboxes, recursive delegation limits, and
  team spawning.
- **MCP support** -- stdio, Streamable HTTP, and legacy SSE transports, with
  automatic tool scanning (including pagination cursors).
- **Skills** -- scans `~/.agents/skills` (plus project and extra paths), gives
  the model a compact preview, and loads full instructions on demand.
- **Image attachments** -- `alt+v` pastes a copied screenshot into the
  message. The image is uploaded through the provider's Files API once and
  referenced by id; images are stored as files, never as base64 in the history.
- **Session toolkit** -- rollback, edit-and-resend, fork, aside questions
  (`/btw`), durable goals, and automatic crash recovery.
- **Single settings browser** -- `/config` exposes thinking display, tool
  display, appearance, reasoning, context, agents, skills, and MCP settings.
- **Command palette** -- `ctrl+p` opens every command in a searchable list,
  and `ctrl+q` quits.
- **Clipboard and proxy support** -- pasting works on every platform and in
  every input field, and HTTP or SOCKS5 proxies are honoured everywhere.

## Requirements

- Python 3.14 or newer
- Windows 10/11 is the primary target (virtual-terminal input, `taskkill`
  process-tree handling). POSIX terminals work through the `termios` path.

## Installation

```bash
git clone https://github.com/your-username/adcode.git
cd adcode
pip install -e .
```

Or with uv:

```bash
uv sync
```

Run it:

```bash
adcode
# or
python main.py
```

## First Run

AdCode opens a four-step provider setup:

1. **Base URL** -- the API endpoint. End with `#` to use it exactly as typed;
   otherwise `/v1` is appended when missing.
2. **API key** -- encrypted with Fernet; the Fernet key lives in the OS
   keyring (Windows Credential Manager, macOS Keychain, or Secret Service).
   An empty key is allowed for local servers.
3. **Display name** -- how the provider appears in the UI.
4. **API structure** -- OpenAI Chat Completions, OpenAI Responses, or
   Anthropic Messages.

The model list is fetched next. If the provider exposes `/models`, a
filterable multi-select table appears (`space` toggles, `enter` confirms, `+`
adds a custom id). If fetching fails, you can retry, probe with another wire,
or enter model ids manually. Metadata is matched against models.dev when
available. A final step lets you review the selected model's context window,
maximum output, vision support, tool calling, and reasoning efforts.

## Keyboard

| Key | Action |
|---|---|
| `enter` | Send the message / run the selected command |
| `shift+enter`, `alt+enter`, `ctrl+j` | Insert a newline |
| trailing `\` + `enter` | Insert a newline (works in every terminal) |
| `up` / `down` | Prompt history, or navigate the `/` command palette |
| `tab` | Complete the selected command |
| `esc` | Interrupt the current turn, or close the palette/overlay |
| `ctrl+c` | Copy the selected transcript text, else the input selection, else clear the input, else interrupt; press twice on an empty prompt to exit |
| `ctrl+v`, `shift+insert` | Paste text, or attach a copied screenshot when the clipboard has no text |
| `alt+v` | Attach the clipboard image, image path, or image URL to the message |
| `ctrl+a` | Select all |
| `shift+left` / `shift+right` | Extend the selection by one character |
| `shift+home` / `shift+end` | Extend the selection to the line edge |
| `ctrl+x` | Cut selection |
| `ctrl+z` / `ctrl+y` | Undo / redo input edits |
| `pgup` / `pgdn`, mouse wheel | Scroll the transcript, or the body of an open dialog |
| mouse drag | Select text in the transcript |
| `home` / `end` | Line start / end |
| `ctrl+home` / `ctrl+end` | Input start / end |
| `ctrl+l` | Redraw the screen |
| `ctrl+p` | Open the command palette (type to filter, enter to run) |
| `ctrl+,` | Open the settings browser |
| `ctrl+h` | Browse and load saved sessions |
| `ctrl+q` | Quit |

Every application shortcut above is rebindable: `/keys` opens the bindings
panel listing each action with its current key. `enter` starts a rebind (press
the new chord; `esc` cancels), `r` resets an action to its default, and a key
already used by another action can take it over (the other action becomes
unbound). Bindings are stored in `~/.adcode/config.json` under `keys` and take
effect immediately. `enter`, `esc`, and the dialog navigation keys stay fixed;
`/help` always shows the live bindings.

### Chords on punctuation

`ctrl+,` and the other chords on punctuation (`ctrl+.`, `ctrl+/`, ...) have no
control byte in the legacy terminal encoding, so a terminal cannot report
them the old way. AdCode asks for the
[kitty keyboard protocol](https://sw.kovidgoyal.net/kitty/keyboard-protocol/)
(`CSI > 1 u`, the "disambiguate escape codes" flag) when it starts, and
terminals that speak it (kitty, Ghostty, WezTerm, alacritty, foot, iTerm2,
Microsoft Terminal, rio, Warp, xterm.js) deliver the chord. Terminals that do
not support the protocol ignore the request and keep sending plain
characters, so on those the chord does not fire -- rebind it with `/keys` if
you need it there.

## Selecting text

Drag with the mouse over the transcript to select text; `ctrl+c` copies the
selection to the clipboard. The selection follows the content, so scrolling
does not move it, and the footer shows how much is selected. Clicking outside
the transcript clears it.

## Command palette

`ctrl+p` opens the command palette: type to narrow the list, `↑`/`↓` to move,
`enter` to run the command and `esc` to close. It lists the same commands the
`/` prefix and the `/help` output show.

While a dialog is open it owns the keyboard and the wheel: keys filter or edit
inside the dialog, and the wheel moves the dialog's list instead of scrolling
the transcript behind it.  A message dialog (the `/help` output, a long
explanation) has no list, so the wheel, the arrow keys and PageUp/PageDown
scroll its text; the panel grows a scrollbar and a hint as soon as the message
is taller than the panel.

## Commands

Type `/` to open the command palette. Up/Down selects a candidate, Tab or
Enter completes it, and Enter runs the command.

| Command | Description |
|---|---|
| `/help` | Commands and key bindings |
| `/config` | Open the settings browser |
| `/keys` | Open the key bindings panel |
| `/model [list\|edit]` | Switch models or edit the selected model's capabilities |
| `/provider add\|edit\|remove\|list` | Manage providers and their models |
| `/effort [level]` | Set `low`, `medium`, `high`, `xhigh`, `max`, or `ultra` |
| `/thinking [mode]` | Set thinking display: `hidden`, `label`, `partial`, `full`, `summary` |
| `/steps [level]` | Preset the work-step settings: `concise`, `standard`, `detailed` |
| `/hold [on\|off\|<retries>]` | Keep retrying a request that was cut short |
| `/max_tokens [n]` | Set the maximum output tokens per request |
| `/compact` | Compact older turns immediately |
| `/context`, `/cache`, `/prefix` | Inspect context size, cache usage, and the stable prefix |
| `/goal <objective>` | Start a durable goal (`status`, `list`, `pause`, `resume`, `complete`, `blocked`, `cancel`) |
| `/agents list\|max\|depth\|wait\|interrupt\|types\|teams` | Inspect and configure sub-agents |
| `/mcp add\|edit\|remove\|list\|doctor\|refresh` | Manage MCP servers and rescan their tools |
| `/skills [list\|reload]` | Browse and reload installed skills |
| `/image <path\|url>` | Attach an image file or URL to the next message |
| `/diff [path]` | Show the Git working-tree diff in the transcript |
| `/sessions` | Browse and load saved sessions |
| `/save [name]` / `/load [name]` | Save / load a session |
| `/rewind` | Roll the conversation back to an earlier turn |
| `/edit` | Rewind to a turn and prefill its message for editing |
| `/fork [name]` | Fork the conversation into a new session |
| `/btw [question]` | Ask a side question without touching the main task; with no question, toggles aside mode for the next message |
| `/clear` or `/new` | Clear the conversation |
| `/theme` | Change the theme or accent colour |
| `/quit` or `/exit` | Exit |

## Settings (`/config`)

`/config` is a tree of grouped settings rendered as a modal panel:

- **Move** with Up/Down.
- **Enter** expands a setting vertically, selects an option, or starts inline
  editing for numbers and text.
- **Esc** cancels the item: it restores the values captured when the item was
  expanded, then collapses it. At the top level it closes the panel.
- **Enter on an expanded setting's title** confirms it and collapses it.
- **Ctrl+S** saves and closes.

Selecting an option that has children expands its sub-settings automatically.
Changes apply to the running app immediately (theme and accent included) and
are persisted to `config.json`.

### Thinking display

| Mode | Behaviour |
|---|---|
| `hidden` | Accept reasoning deltas silently |
| `label` | One label line with a fixed or rotating suffix |
| `partial` | A configurable window of the rendered reasoning |
| `full` | The complete reasoning stream |
| `summary` | The reasoning is kept but not spelled out: one line per finished thought, `thought of 412 tokens` |

The `label` mode configures the label text (`Thinking`, `Reasoning`,
`Polishing`, ...), the suffix animation (`fixed` or `rotating`), the rotation
frames, and the per-frame duration. The `partial` mode configures:

- two negative visual-row indices (for example `-5` to `-1` shows the last
  five visual rows; `-6` to `-2` ends at the second-to-last row),
- the display style (`normal` Markdown or `gradient` with a configurable top
  row alpha from 0 to 255),
- the omitted-content marker (`none`, an ellipsis `label`, or a character
  `count`).

### Frame rate

The app draws at 60 frames per second by default. `/config` exposes
`Appearance > Frames per second` (`ui.fps`, 10 to 240) to change it. A frame
that has nothing new to show writes no bytes to the terminal, so a higher
rate costs nothing while the app is idle.

### Interruptions and `/hold`

A turn can stop for two very different reasons: the user pressed `esc`, or
something went wrong on the way (a provider error, a dropped connection, a
response that ended without a finish signal).  AdCode treats them
differently.

- **`esc` reaches the provider.**  The stream stops at the next chunk instead
  of running to the end of the response, so an interrupt is immediate.
- **The work that was done stays in the history.**  The reasoning of a round
  that never finished is kept as an assistant turn that says so, together
  with the partial answer.  Without it the next turn starts from a blank
  slate: the model re-reads the same files and re-plans the same work.
- **`/hold` retries the request.**  With `/hold on` a request whose stream
  broke is re-sent until it works.  The only things that stop the loop are
  an explicit `esc` and a response that actually finished; when the retries
  run out, AdCode gives up and says so.

| Command | Effect |
|---|---|
| `/hold` | Toggle the mode |
| `/hold on` / `/hold off` | Turn it on or off |
| `/hold <n>` | Set how many times a broken request is retried (default 20) |

The settings live in `/config` under **Hold** (retry count and the delay
between attempts).

### Work steps

The thinking and the tool settings live in one **Work steps** group of
`/config`; they decide how much of the work chain the transcript spells out.

Two of them do the heavy lifting:

- **Chain of calls** (`tools.chain`) -- `separate` gives every call its own
  row, `merged` folds a chain of consecutive calls into one row with the
  tool names and their counts: `bash x2, grep, write`.  A merged chain shows
  no details; the whole chain stays in the history.
- **Thinking display** (`thinking.mode`) -- `summary` keeps the reasoning but
  only says how much there was: `thought of 412 tokens`.

`/steps` is a shortcut that writes those settings:

| Level | Writes |
|---|---|
| `concise` | `thinking.mode = summary`, `tools.chain = merged` |
| `standard` | `thinking.mode = partial`, `tools.chain = separate` |
| `detailed` | `thinking.mode = full`, `tools.chain = separate`, `tools.result_mode = full` |

### The status line

The line under the input carries the state of the session.  `/config` has a
**Footer status** group for it; almost every part is configurable.

**While the agent works.** A light runs along the line: it enters at the left
edge, crosses the screen once per second and leaves at the right one.  The
wait for the provider (`connecting`) has no light.

**A command that reports its progress.**  When the running command prints a
percentage -- a pytest run, a build, a download -- the line becomes a
progress bar: the background fills up to the percentage and the text names
the command and the number (`pytest 45%`).

A percentage on its own means nothing: pip prints one per package, a build
tool one per file.  Only a value that keeps climbing is a total progress, so
the samples are compared -- a value that jumps around is not one, and the
light comes back instead of a bar that lies.

**When the task is done.**  The line turns green with white text: the green
runs in from the left edge in half a second, stays until the user types
something, and then leaves again from the left to the right in half a
second.

| Setting | Meaning |
|---|---|
| `footer.sweep_enabled` | The light that runs along the line |
| `footer.sweep_period` | Seconds per crossing |
| `footer.sweep_span` | The width of the light, in cells |
| `footer.progress_enabled` | The progress bar |
| `footer.progress_window` | How long a percentage stays fresh |
| `footer.progress_samples` | Samples needed before the bar shows |
| `footer.done_enabled` | The green state of a finished task |
| `footer.done_wipe` | Seconds for the wipe in and out |
| `footer.done_hold` | Stay green until the user types |
| `footer.done_color` | The green (`#rrggbb`, empty = the palette green) |
| `footer.done_text_color` | The font (`#rrggbb`, empty = white) |

### Tool display

| Mode | Behaviour |
|---|---|
| `hidden` | Execute silently |
| `call` | One compact line: `│ tool  args` on a light panel background; optionally show arguments |
| `result` | Only the result section |
| `both` | The call line and the result section |

The call line bleeds one cell into the transcript's left margin, and its accent
bar is a left-edge block (`▏`, one cell wide, same room as `│`) rather than a
centred glyph, so the rule really sits flush with the screen edge while the
gutter keeps the name one space away from it. Result rows stay aligned with the
wrapped text.

A chain of consecutive calls can be folded into one row (`tools.chain`,
see **Work steps**): `▏ bash x2, grep, write`.  The row names the tools the
chain used and how often; the arguments and the output stay in the history.

Result sections support the same negative-row window, style, gradient, and
ellipsis controls as thinking, plus a full mode with an optional line cap.
File diffs are rendered inline and can be toggled.

## Configuration File

`config.json` lives in `~/.adcode/` (user-level, shared across projects). API
keys are encrypted before writing. An older file next to the package is moved
there automatically on first load. A minimal file looks like this:

```json
{
    "version": 2,
    "providers": [
        {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "<encrypted>",
            "wire": "openai chat completions",
            "models": [
                {
                    "id": "deepseek-chat",
                    "context_window": 131072,
                    "max_output_tokens": 8192,
                    "reasoning": false,
                    "tool_call": true,
                    "source": "models.dev"
                }
            ]
        }
    ],
    "selected_model": "DeepSeek::deepseek-chat",
    "thinking": {"mode": "partial", "style": "gradient"},
    "tools": {"mode": "both", "chain": "separate", "result_mode": "partial"},
    "hold": {"enabled": false, "max_retries": 20, "delay": 2},
    "footer": {"sweep_enabled": true, "sweep_period": 1, "sweep_span": 12,
               "progress_enabled": true, "done_enabled": true,
               "done_wipe": 0.5, "done_hold": true},
    "agents": {"enabled": true, "max_concurrent": 10, "max_depth": 4},
    "agent_types": {},
    "agent_teams": {}
}
```

`selected_model` uses the `provider::model` form. `agent_types` and
`agent_teams` default to the built-in sets (general, explorer, reviewer,
worker; explore, build, review) and can be replaced with custom definitions:

```json
{
    "agent_types": {
        "docs": {
            "description": "Writes and reviews documentation",
            "system": "You are a documentation specialist.",
            "tools": ["read", "write", "edit", "glob", "grep"],
            "max_turns": 20
        }
    },
    "agent_teams": {
        "ship": ["docs", "reviewer"]
    }
}
```

## MCP

MCP servers are managed by `/mcp` and stored in `~/.adcode/config.json`:

```json
{
    "name": "filesystem",
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
}
```

```json
{
    "name": "remote",
    "transport": "http",
    "url": "https://example.com/mcp"
}
```

Transport values are `stdio`, `http` (Streamable HTTP), and `sse` (legacy
HTTP event stream). Servers connect in parallel, expose tools under
`mcp__<server>__<tool>` names, and paginate `tools/list` automatically.
`/mcp doctor` shows protocol versions, tool counts, errors, and recent stderr;
`/mcp refresh` rescans every connected server.

## Skills

Skills are directories containing `SKILL.md` with YAML frontmatter:

```markdown
---
name: my-skill
description: One line that tells the model when to use this skill.
---

Full instructions...
```

AdCode scans `~/.agents/skills`, `<cwd>/.agents/skills`,
`<cwd>/.adcode/skills`, and any `skills.extra_paths` entries. The system
prompt lists each skill's name and description; the model calls
`skill_list(query)` to search and `skill_load(name)` to read the full
instructions and see the skill's supporting files. `/skills reload` rescans
after adding or editing a skill.

## Sessions

Sessions are stored under `sessions/` as JSON (history, transcript, compaction
archive, and prefix generation). Every turn updates a `__last__` checkpoint so
a crash or restart loses at most the last checkpoint interval. Old AdCode 1.x
session files are migrated on load.

- `/rewind` lists user turns; selecting one drops that turn and everything
  after it.
- `/edit` does the same but prefills the selected message so you can change it
  and resend.
- `/fork` copies the conversation into a new session and continues there.
- `/save` / `/load` / `/sessions` manage named sessions.

Goals live in `sessions/goals.json` and are injected into the system prompt
while active. Global instructions are read from `~/.adcode/AGENTS.md` and
project instructions from `AGENTS.md` (or `prompt`) in the working directory.
Both are inserted into the system prompt; when they conflict, the project
instructions take precedence.

The system prompt also carries an `<identity>` block telling the model who it
is: the name AdCode, the model id of the currently selected model, and the
fact that it runs on the AICF harness. The model id follows model changes,
and a sub-agent running a different model gets its own identity block
instead of the parent's.

## Images

Copy a screenshot, then press `alt+v` (or use `/image <path|url>`). The image
is stored under `sessions/images/` with a content-addressed name and a short
`[image: sessions/images/paste-<hash>.png]` marker is inserted into the prompt.
The marker is plain text, so it survives session save/load, rewind, edit and
fork, and it costs almost nothing in the history.

`alt+v` also works when the clipboard holds a file path, a `file://` URI or an
image link instead of pixels (a file copied in Explorer or Finder lands that
way). Attaching prints the stored name and size, and warns when the selected
model is not marked as vision-capable. Each attachment counts in the footer as
`N images` until the message is sent.

The marker is expanded into a native image part only while a request is being
built, and the image is first uploaded through the provider's Files API:

- **Upload once** -- `POST /files` through the SDK, then the returned id is
  cached in `sessions/images/files.json` under `provider|base_url`, so the same
  screenshot is uploaded a single time no matter how many turns mention it.
- **OpenAI Responses** -- `input_text` and `input_image` parts; the image part
  carries `file_id`.
- **Anthropic Messages** -- `text` and `image` blocks with a
  `{"type": "file", "file_id": ...}` source.
- **OpenAI Chat Completions** -- this wire has no file id, so the part carries
  a data URL. That is the only wire where base64 reaches the request body.

An endpoint that rejects the upload (a 400/403/404 and friends, typical of
third-party OpenAI-compatible servers) is recorded as unsupported and the image
is inlined instead: `image_url` data URL for Responses, and a `Path` handed to
the Anthropic SDK so it reads and encodes the bytes while serializing. AdCode
never builds the base64 string itself. A 5xx or a network error is not
remembered, so the next turn tries again.

A file id is not immortal: uploads expire and some endpoints only accept the
inline shape. When a request that carried a file id comes back rejected
(400/403/404/422), the id is dropped, the request is retried once with the image
inlined, and that provider stops using file ids from then on. Nothing is lost
but one round trip.

Base64 is therefore transient request data: it is never written to the session
JSON, never shown in the transcript and never counted as context text. Each
marker is estimated at about 1,200 tokens for compaction. `http(s)` URLs can be
used instead of a local file and are passed straight through as image URLs, with
no upload and no encoding. Providers that do not accept images will simply
reject the request; models from models.dev are marked with vision support when
known.

### The model looks on its own

A vision model does not have to wait for a paste: `view_image` takes a path or
an `http(s)` URL and answers with a one-line description plus the same marker.
The file is copied into `sessions/images/` first (content-addressed, so viewing
the same screenshot twice stores it once), which keeps the conversation valid
after the original is edited or deleted.

Tool results are plain text on all three wires, so the picture cannot ride
inside them. Instead the loop stages a `<tool_images>` user block right after
the tool results:

```
assistant: tool_calls [view_image("shot.png")]
tool:      Loaded shot.png (1920x1080 image/png, 240 KB). ...
user:      <tool_images>
           [image: sessions/images/paste-<hash>.png]
           </tool_images>
```

That block is skipped by rewind and edit (`turn_starts` ignores the `<tool_`
prefix), so it never appears as a user turn, and it reaches every wire as a
normal image part. `view_image` is only offered to models marked
vision-capable: a text-only model never sees the schema, and sub-agents inherit
the same rule from their own model.

## Rendering Architecture

```
widgets ──draw──▶ Canvas (width × height cells, fg/bg/attrs/alpha)
                      │
              alpha compositing
                      │
              full-frame ANSI string ──▶ TerminalScreen (WriteConsoleW / UTF-8)
```

- `adcode/tui/canvas.py` holds the cell buffer and primitives (`draw_text`,
  `fill_rect`, `box`, `hline`, `vline`) with clipping, wide-glyph handling,
  and alpha blending.
- `adcode/tui/screen.py` enters the alternate buffer, enables bracketed paste
  and SGR mouse reporting, and writes each frame once.
- `adcode/tui/app.py` owns the layout (header, virtual transcript viewport,
  prompt, footer), scrolling, input handling, and overlays.
- `adcode/tui/transcript.py` renders transcript items and live streaming
  state according to the `/config` display policies.
- `adcode/tui/markdown.py` is an incremental Markdown renderer producing
  styled spans (headings, emphasis, code fences, lists, quotes, tables, links,
  math).

## Agent Loop

`adcode/agent_loop.py` runs on a worker thread. It stages user messages,
streams one request at a time, executes tool calls, feeds results back, drains
sub-agent mailboxes, compacts the context when needed, and persists progress.
The UI consumes a thread-safe event queue and renders the live state.

- Interrupts are cooperative: `esc` sets an event checked between stream
  events; the shell tool polls it and kills the process tree.
- `finish_reason == "length"` triggers a bounded continuation request so long
  reasoning chains are not truncated.
- Compaction keeps the most recent turns, summarizes the rest with a streaming
  model call, appends a deterministic fallback when the model call fails, and
  archives the removed messages.
- An interrupted turn keeps the work it finished: the tool rounds that
  completed stay in the history, the reasoning of the round that was cut off
  stays with it, and the sub-agent updates a failed turn never delivered go
  back into the mailbox for the next turn.
- `/hold` re-sends a request whose stream broke (a provider error, a dropped
  connection, a response without a finish signal) until it works, the user
  interrupts, or the retries run out.

### Wire shapes

Every wire needs its own message shape, and a tool round must reach the model
as a pair it can follow:

| Wire | Assistant turn | Tool result |
|---|---|---|
| OpenAI Chat Completions | `{"role": "assistant", "tool_calls": [...]}` | `{"role": "tool", "tool_call_id": ..., "content": ...}` |
| OpenAI Responses | `{"type": "function_call", "call_id": ..., ...}` | `{"type": "function_call_output", "call_id": ..., "output": ...}` |
| Anthropic Messages | a `tool_use` content block | a `tool_result` content block |

The Responses wire takes *items*, not chat messages.  Sending the chat shape
there leaves the model without a result it can attach to its own call: it
stops learning from the tools and repeats them until the turn is interrupted.
`adcode/llm/client.py` therefore translates the history per wire.

## Project Structure

```
adcode/
├── cli.py            # console entry point
├── net.py            # proxy-aware HTTP (SOCKS5, urllib fallbacks)
├── agent_loop.py     # the engine facade: events, sessions, control plane
├── engine.py         # the engine itself: interruptions, hold, reasoning
├── config.py         # settings schema, migration, atomic save
├── crypto.py         # Fernet API-key encryption
├── models_dev.py     # models.dev catalogue and cache
├── session.py        # transcript items, save/load, rewind/fork
├── goals.py          # durable goal store
├── images.py         # image markers, file store, clipboard, per-wire helpers
├── skills.py         # skill discovery, preview and tools
├── system_prompt.py  # global/project instructions, environment, goals and skills
├── commands.py       # slash commands and completion
├── diff.py           # file snapshots, unified diffs, git worktree
├── theme.py          # palettes and semantic colour roles
├── llm/
│   ├── client.py     # OpenAI Chat/Responses and Anthropic streaming
│   ├── context.py    # conversation state, stable prefix, compaction
│   └── usage.py      # token estimates, budget, usage normalization
├── tools/
│   ├── builtin.py    # read, write, edit, glob, grep, bash, fetch, diff, ls
│   ├── registry.py   # schemas, dispatch, diff attachment
│   └── runtime.py    # per-thread tool context (interrupts, live output)
├── mcp/
│   ├── client.py     # stdio, Streamable HTTP, and SSE transports
│   └── manager.py    # server lifecycle, tool registration, diagnostics
├── agents/
│   └── manager.py    # sub-agent tree, teams, mailboxes, ReAct workers
└── tui/
    ├── app.py        # layout, input, scrolling, overlays
    ├── canvas.py     # software renderer and alpha compositing
    ├── screen.py     # terminal modes and frame output
    ├── input.py      # cross-platform key/paste/mouse parser
    ├── clipboard.py  # clipboard access and the paste bindings
    ├── markdown.py   # incremental Markdown -> styled spans
    ├── text.py       # spans and width-aware wrapping
    ├── transcript.py # transcript and live-state rendering
    ├── dialogs.py    # select, multi-select, prompts, confirmations
    ├── settings.py   # the /config tree browser
    └── provider_ui.py# provider setup and model editor
```

## Development

```bash
uv sync
python -m pytest
```

The test suite covers canvas compositing, wrapping and Markdown, config
migration, models.dev matching, compaction, sessions and rewind, skills, the
tools, the agent loop, sub-agents, all three MCP transports, commands, input
parsing, transcript display policies, settings behavior, and prompt layout.

It also contains real controlled-run simulations: the actual application is
started against a temporary config, a local fake LLM gateway, and scripted
keyboard/mouse input.  See [Test Harness](#test-harness).

## Test Harness

The harness starts the real `AdCodeApp` loop on a worker thread and replaces
only the terminal edges:

- **Temporary config and workspace** -- every run gets a private
  `project/` and `home/.adcode/config.json`; the real `~/.adcode` is never
  read or written.
- **Fake local LLM gateway** -- a real HTTP server that speaks OpenAI Chat
  Completions, OpenAI Responses and Anthropic Messages over SSE.  The
  production `LLMClient` and the official SDKs make the request, accumulate
  tool-call deltas and parse usage exactly as they do in production.
- **Scripted input** -- raw arrow/control/paste/SGR-mouse sequences are fed
  through AdCode's own `KeyParser`, so the app's real key handlers run.
- **Recorded frames** -- a fake screen snapshots every composited frame,
  replays the serialized ANSI through a small VT emulator and exposes the
  text, cells, colours and cursor.

Automated simulations live in `tests/test_simulation_*.py`:

```bash
python -m pytest tests/test_simulation_chat.py tests/test_simulation_input.py \
                 tests/test_simulation_config.py tests/test_simulation_gateway.py
```

The manual toolkit produces screenshots and run data:

```bash
python -m bench list                 # scenarios
python -m bench run chat             # one scenario
python -m bench run all              # every scenario
python -m bench run commands --width 120 --height 40 --out out/
```

Each run writes:

```
out/
├── report.md               # human-readable summary with images
├── manifest.json           # environment, state, frame index, warnings
├── transcript.json         # every TranscriptItem
├── gateway_requests.json   # exact request bodies the app sent
├── config.json             # the temporary config that was used
├── session.json            # persisted session payload
├── workspace/              # copy of the simulated project and sessions
└── screens/
    ├── 001-idle.json       # full cell dump
    ├── 001-idle.svg        # dependency-free vector screenshot
    ├── 001-idle.png        # raster screenshot (Pillow)
    └── 001-idle.txt        # plain text as the terminal would show it
```

SVG screenshots always work.  PNG needs Pillow; if the app virtualenv does not
have it, the harness automatically looks for an interpreter that does
(`ADCODE_RASTER_PYTHON`, then `python`/`python3`/`py`).  Scenarios: `chat`,
`tool-call`, `keyboard`, `commands`, `scroll`, `resize`, `interrupt`.

## Clipboard

`ctrl+v` and `shift+insert` paste the clipboard into the prompt, into the
settings editor, into the provider wizard and into every dialog that takes
text (the model list filter, the manual model id field, the text prompts).
Terminals that send a bracketed paste are supported as well.

The clipboard is read through the first mechanism that works:

| Platform | Mechanisms |
|---|---|
| Windows | `pyperclip`, PowerShell `Get-Clipboard` |
| macOS | `pyperclip`, `pbpaste` |
| Linux | `pyperclip`, `wl-paste`, `xclip`, `xsel`, Tk |

A copy made inside AdCode is also kept in memory, so pasting it back works
even on a machine without any clipboard tool. When the clipboard holds no
text, `ctrl+v` falls back to attaching the clipboard image, and `alt+v`
attaches it directly.

## Proxies

`HTTP_PROXY`, `HTTPS_PROXY` and `ALL_PROXY` are honoured for the model APIs,
the models.dev catalogue, the `fetch` tool and the HTTP/SSE MCP transports.
SOCKS proxies (`socks5://` and `socks5h://`) are supported as well; the
`httpx2[socks]` extra is installed with the project, so the SOCKS transport
is always available. A `socks://` value is treated as `socks5://`, and an
unsupported scheme is reported with a message that names the problem instead
of crashing the app.

## Dependencies

- `openai` -- OpenAI-compatible Chat Completions and Responses clients
- `anthropic` -- Messages API client
- `httpx2[socks]` -- HTTP transport, including the SOCKS5 proxy support
- `cryptography` -- Fernet encryption for API keys
- `keyring` -- OS credential storage
- `pyperclip` -- clipboard access

## License

MIT

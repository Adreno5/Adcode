/* AdCode browser client: paints the cell grid the server streams and turns
   taps and keys into terminal input.  No framework and no bundler: the page
   has to work offline on a phone that can only reach this machine. */
(function () {
  'use strict';

  var screenEl = document.getElementById('screen');
  var gridEl = document.getElementById('grid');
  var caretEl = document.getElementById('caret');
  var keybarEl = document.getElementById('keybar');
  var tapEl = document.getElementById('tap');
  var bannerEl = document.getElementById('banner');

  var params = new URLSearchParams(window.location.search);
  var TOKEN = params.get('token') || '';

  /* Geometry.  A character cell is charRatio em wide and LINE_FACTOR em tall,
     measured from the browser's own font, so the grid a window asks for is the
     grid it will really get. */
  var TARGET_FONT = 13;
  var LINE_FACTOR = 1.3;
  var MIN_FONT = 4;
  var MAX_FONT = 26;
  /* The app lays itself out in columns; a phone in portrait should shrink the
     font rather than ask for a 45-column terminal that wraps everything. */
  var MIN_COLS = 64;
  var MIN_ROWS = 12;
  var MODIFIERS = ['ctrl', 'alt', 'shift'];
  var WHEEL_STEP = 24;

  var socket = null;
  var reconnectDelay = 500;
  var cols = 100;
  var rows = 30;
  var rowEls = [];
  var caret = { x: 0, y: 0, show: false };
  var advance = 8;
  var lineHeight = 16;
  var charRatio = 0.6;
  var mods = { ctrl: false, alt: false, shift: false };
  var composing = false;
  var altHeld = false;
  var touch = { active: false, y: 0, scroll: 0, moved: false };

  /* -- measuring and fitting ------------------------------------ */

  function measureRatio() {
    var probe = document.createElement('span');
    probe.style.position = 'absolute';
    probe.style.visibility = 'hidden';
    probe.style.whiteSpace = 'pre';
    probe.style.fontSize = '100px';
    probe.style.fontFamily = 'inherit';
    probe.textContent = new Array(101).join('M');
    document.body.appendChild(probe);
    var width = probe.getBoundingClientRect().width;
    document.body.removeChild(probe);
    if (width > 0) charRatio = width / 100 / 100;
  }

  function available() {
    var style = window.getComputedStyle(screenEl);
    var padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    var padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    return {
      width: Math.max(80, screenEl.clientWidth - padX),
      height: Math.max(60, screenEl.clientHeight - padY)
    };
  }

  /* The grid this window would like: as many cells as fit at the target font.
     The server keeps every connected window on the smallest grid that was
     asked for, so no device has to scroll sideways to see the screen. */
  function preferredGrid() {
    var box = available();
    var fitCols = Math.floor(box.width / (TARGET_FONT * charRatio));
    var cols = Math.max(MIN_COLS, Math.min(400, fitCols));
    /* The font the grid will end up with, so forcing more columns than the
       target font allows shrinks the text instead of cropping rows. */
    var fontSize = box.width / (cols * charRatio);
    var rows = Math.floor(box.height / (fontSize * LINE_FACTOR));
    return {
      cols: cols,
      rows: Math.max(MIN_ROWS, Math.min(200, rows))
    };
  }

  function fit() {
    var box = available();
    var byWidth = box.width / Math.max(1, cols * charRatio);
    var byHeight = box.height / Math.max(1, rows * LINE_FACTOR);
    var size = Math.min(byWidth, byHeight);
    size = Math.max(MIN_FONT, Math.min(MAX_FONT, size));
    var fontSize = Math.floor(size * 100) / 100;
    advance = fontSize * charRatio;
    lineHeight = fontSize * LINE_FACTOR;
    var root = document.documentElement.style;
    root.setProperty('--fs', fontSize + 'px');
    root.setProperty('--lh', lineHeight + 'px');
    gridEl.style.width = (cols * advance) + 'px';
    placeCaret();
  }

  function ensureRows(count) {
    while (rowEls.length < count) {
      var el = document.createElement('div');
      el.className = 'row';
      gridEl.appendChild(el);
      rowEls.push(el);
    }
    while (rowEls.length > count) {
      var last = rowEls.pop();
      last.remove();
    }
  }

  function setGrid(newCols, newRows) {
    if (!newCols || !newRows) return;
    if (newCols === cols && newRows === rows && rowEls.length === rows) return;
    cols = newCols;
    rows = newRows;
    ensureRows(rows);
    fit();
  }

  /* -- painting ------------------------------------------------- */

  function hex(value) {
    var text = (value >>> 0).toString(16);
    return '#' + '000000'.slice(text.length) + text;
  }

  function paint(y, runs) {
    var el = rowEls[y];
    if (!el || !runs) return;
    var fragment = document.createDocumentFragment();
    for (var i = 0; i < runs.length; i++) {
      var run = runs[i];
      var span = document.createElement('span');
      span.textContent = run[0];
      span.style.color = hex(run[1]);
      span.style.backgroundColor = hex(run[2]);
      var attrs = run[3];
      if (attrs & 1) span.style.fontWeight = '700';
      if (attrs & 2) span.style.opacity = '0.7';
      if (attrs & 4) span.style.fontStyle = 'italic';
      if (attrs & 8) span.style.textDecoration = 'underline';
      if (attrs & 16) span.style.textDecoration = 'line-through';
      fragment.appendChild(span);
    }
    el.replaceChildren(fragment);
  }

  function placeCaret() {
    caretEl.style.display = caret.show ? 'block' : 'none';
    caretEl.style.transform = 'translate(' + (caret.x * advance) + 'px,'
      + (caret.y * lineHeight) + 'px)';
  }

  function applyCaret(message) {
    if (message.caret) {
      caret.x = message.caret[0];
      caret.y = message.caret[1];
    }
    if (typeof message.show === 'boolean') caret.show = message.show;
    placeCaret();
  }

  /* -- connection ----------------------------------------------- */

  function send(payload) {
    if (!socket || socket.readyState !== 1) return;
    socket.send(JSON.stringify(payload));
  }

  function sendSize() {
    var grid = preferredGrid();
    send({ t: 'size', cols: grid.cols, rows: grid.rows });
  }

  function showBanner(text) {
    var wasHidden = bannerEl.hidden;
    if (!text) {
      bannerEl.hidden = true;
      bannerEl.textContent = '';
    } else {
      bannerEl.hidden = false;
      bannerEl.textContent = text;
    }
    /* The banner takes a row away from the terminal, so the grid has to be
       measured again when it appears or goes away. */
    if (wasHidden !== bannerEl.hidden) fit();
  }

  function onMessage(event) {
    var message;
    try {
      message = JSON.parse(event.data);
    } catch (error) {
      return;
    }
    if (message.t === 'full') {
      showBanner('');
      setGrid(message.w, message.h);
      ensureRows(rows);
      for (var i = 0; i < rowEls.length; i++) rowEls[i].textContent = '';
      var lines = message.lines || [];
      for (var j = 0; j < lines.length; j++) paint(lines[j][0], lines[j][1]);
      applyCaret(message);
      if (message.title) document.title = message.title;
    } else if (message.t === 'rows') {
      showBanner('');
      var changed = message.lines || [];
      for (var k = 0; k < changed.length; k++) {
        paint(changed[k][0], changed[k][1]);
      }
      applyCaret(message);
    } else if (message.t === 'size') {
      setGrid(message.w, message.h);
    } else if (message.t === 'title') {
      document.title = message.v;
    } else if (message.t === 'bell') {
      /* nothing to ring in a browser */
    } else if (message.t === 'error' || message.t === 'closed') {
      showBanner(message.v || 'the session ended');
    }
  }

  function connect() {
    var scheme = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    var url = scheme + window.location.host + '/ws'
      + (TOKEN ? '?token=' + encodeURIComponent(TOKEN) : '');
    var ws = new WebSocket(url);
    socket = ws;
    ws.onopen = function () {
      reconnectDelay = 500;
      showBanner('');
      sendSize();
    };
    ws.onmessage = onMessage;
    ws.onclose = function () {
      if (socket === ws) socket = null;
      showBanner('connecting to AdCode...');
      window.setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(8000, reconnectDelay * 2);
    };
    ws.onerror = function () {
      try { ws.close(); } catch (error) {}
    };
  }

  /* -- key input ------------------------------------------------ */

  function canonical(list, base) {
    var ordered = [];
    for (var i = 0; i < MODIFIERS.length; i++) {
      if (list.indexOf(MODIFIERS[i]) !== -1) ordered.push(MODIFIERS[i]);
    }
    ordered.push(base);
    return ordered.join('+');
  }

  function combine(name) {
    var parts = name.split('+');
    var base = parts.pop();
    for (var i = 0; i < MODIFIERS.length; i++) {
      var mod = MODIFIERS[i];
      if (mods[mod] && parts.indexOf(mod) === -1) parts.push(mod);
    }
    return canonical(parts, base);
  }

  function clearMods() {
    for (var i = 0; i < MODIFIERS.length; i++) mods[MODIFIERS[i]] = false;
    var buttons = keybarEl.querySelectorAll('button.mod');
    for (var j = 0; j < buttons.length; j++) {
      buttons[j].classList.remove('on');
    }
  }

  function sendKey(name) {
    send({ t: 'key', k: combine(name) });
    clearMods();
  }

  function sendText(text) {
    if (!text) return;
    if ((mods.ctrl || mods.alt) && text.length === 1) {
      send({ t: 'key', k: combine(text) });
      clearMods();
      return;
    }
    send({ t: 'text', v: text });
    if (mods.ctrl || mods.alt || mods.shift) clearMods();
  }

  function sendPaste(text) {
    if (!text) return;
    send({ t: 'paste', v: text });
  }

  /* The clipboard the user copied into is the browser's, so it has priority
     over the machine running AdCode.  The Clipboard API only exists in a
     secure context, and this page is usually plain HTTP on the LAN, so a
     caller can pass a fallback for when it is unavailable. */
  function pasteFromClipboard(fallback) {
    if (navigator.clipboard && navigator.clipboard.readText) {
      navigator.clipboard.readText().then(function (text) {
        if (text) sendPaste(text);
        else if (fallback) fallback();
      }, function () {
        if (fallback) fallback();
        else focusTap();
      });
      return;
    }
    if (fallback) fallback();
    else focusTap();
  }

  function sendWheel(action) {
    send({ t: 'mouse', a: action, x: 1, y: 1 });
  }

  /* A terminal reports whether a modifier is *held*; a touch screen has no
     such state, so the button reports its own press and release.  Holding Alt
     is what reveals every detail in the app. */
  var heldModifiers = {};

  function sendModifier(name, action) {
    send({ t: 'modifier', m: name, a: action });
  }

  function releaseModifiers() {
    for (var name in heldModifiers) {
      if (heldModifiers[name]) {
        heldModifiers[name] = false;
        sendModifier(name, 'release');
      }
    }
  }

  /* Only re-focus when the soft keyboard was already up: a tap on the key bar
     must not pop the keyboard open on someone who closed it on purpose. */
  function keepKeyboard() {
    if (document.activeElement === tapEl) focusTap();
  }

  function focusTap(event) {
    if (event && event.preventDefault) event.preventDefault();
    try { tapEl.focus({ preventScroll: true }); } catch (error) { tapEl.focus(); }
  }

  /* Keys with a name of their own; the server turns each into the sequence a
     terminal sends, so a button here behaves exactly like a real keypress. */
  var KEY_TABLE = [
    { k: 'esc', label: 'Esc', title: 'escape: interrupt or close' },
    { k: 'tab', label: 'Tab', title: 'tab: complete' },
    { k: 'shift+tab', label: '⇤', title: 'shift+tab' },
    { k: 'enter', label: '⏎', title: 'enter: send' },
    { k: 'backspace', label: '⌫', title: 'backspace' },
    { k: 'up', label: '↑', title: 'up: history' },
    { k: 'down', label: '↓', title: 'down: history' },
    { k: 'left', label: '←', title: 'left' },
    { k: 'right', label: '→', title: 'right' },
    { k: 'home', label: 'Home', title: 'line start' },
    { k: 'end', label: 'End', title: 'line end' },
    { k: 'pgup', label: 'PgUp', title: 'scroll the transcript up' },
    { k: 'pgdn', label: 'PgDn', title: 'scroll the transcript down' },
    { k: 'delete', label: 'Del', title: 'delete forward' },
    { k: 'ctrl+c', label: '^C', title: 'ctrl+c: copy, clear, interrupt' },
    { k: 'ctrl+p', label: '^P', title: 'ctrl+p: command palette' },
    { k: 'ctrl+h', label: '^H', title: 'ctrl+h: sessions' },
    { k: 'ctrl+j', label: '^J', title: 'ctrl+j: newline' },
    { k: 'ctrl+a', label: '^A', title: 'ctrl+a: select all' },
    { k: 'ctrl+x', label: '^X', title: 'ctrl+x: cut' },
    { k: 'ctrl+v', label: '^V', title: 'ctrl+v: paste' },
    { k: 'ctrl+q', label: '^Q', title: 'ctrl+q: quit' },
    { k: '/', label: '/', title: 'slash: commands' }
  ];

  function buildKeybar() {
    var fragment = document.createDocumentFragment();
    for (var i = 0; i < MODIFIERS.length; i++) {
      var mod = MODIFIERS[i];
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'mod wide';
      button.textContent = mod === 'ctrl' ? 'Ctrl' : (mod === 'alt' ? 'Alt' : 'Shift');
      button.dataset.mod = mod;
      button.title = 'tap to arm, hold to reveal details';
      button.addEventListener('pointerdown', (function (name) {
        return function () {
          heldModifiers[name] = true;
          sendModifier(name, 'press');
        };
      })(mod));
      button.addEventListener('click', (function (name) {
        return function () {
          mods[name] = !mods[name];
          var buttons = keybarEl.querySelectorAll('button.mod');
          for (var index = 0; index < buttons.length; index++) {
            var other = buttons[index];
            other.classList.toggle('on', !!mods[other.dataset.mod]);
          }
          keepKeyboard();
        };
      })(mod));
      fragment.appendChild(button);
    }
    for (var j = 0; j < KEY_TABLE.length; j++) {
      var entry = KEY_TABLE[j];
      var key = document.createElement('button');
      key.type = 'button';
      key.textContent = entry.label;
      key.title = entry.title || entry.k;
      key.addEventListener('click', (function (name) {
        return function () {
          sendKey(name);
          keepKeyboard();
        };
      })(entry.k));
      fragment.appendChild(key);
    }
    var keyboard = document.createElement('button');
    keyboard.type = 'button';
    keyboard.className = 'wide';
    keyboard.textContent = 'Keyboard';
    keyboard.addEventListener('click', function () { focusTap(); });
    /* mousedown is the one event whose default blurs the textarea without
       also cancelling the click, so the bar keeps the soft keyboard open. */
    fragment.appendChild(keyboard);
    var paste = document.createElement('button');
    paste.type = 'button';
    paste.className = 'wide';
    paste.textContent = 'Paste';
    paste.addEventListener('click', function () {
      pasteFromClipboard(null);
      keepKeyboard();
    });
    fragment.appendChild(paste);
    keybarEl.appendChild(fragment);
    keybarEl.addEventListener('mousedown', function (event) {
      event.preventDefault();
    });
    window.addEventListener('pointerup', releaseModifiers);
    window.addEventListener('pointercancel', releaseModifiers);
    window.addEventListener('blur', releaseModifiers);
  }

  /* -- soft keyboard -------------------------------------------- */

  function bindSoftKeyboard() {
    tapEl.addEventListener('input', function () {
      if (composing) return;
      var value = tapEl.value;
      if (!value) return;
      tapEl.value = '';
      sendText(value);
    });
    tapEl.addEventListener('compositionstart', function () { composing = true; });
    tapEl.addEventListener('compositionend', function () {
      composing = false;
      var value = tapEl.value;
      tapEl.value = '';
      sendText(value);
    });
    tapEl.addEventListener('beforeinput', function (event) {
      var type = event.inputType || '';
      if (type.indexOf('delete') === 0) {
        event.preventDefault();
        sendKey('backspace');
      }
    });
    function setAltHeld(held) {
      if (held === altHeld) return;
      altHeld = held;
      send({ t: 'mod', k: 'alt', a: held ? 'press' : 'release' });
    }

    tapEl.addEventListener('keydown', function (event) {
      if (event.isComposing) return;
      if (event.key === 'Alt') {
        setAltHeld(true);
        return;
      }
      var name = null;
      switch (event.key) {
        case 'Enter': name = event.shiftKey ? 'shift+enter' : 'enter'; break;
        case 'Backspace': name = 'backspace'; break;
        case 'Tab': name = event.shiftKey ? 'shift+tab' : 'tab'; break;
        case 'Escape': name = 'esc'; break;
        case 'ArrowUp': name = 'up'; break;
        case 'ArrowDown': name = 'down'; break;
        case 'ArrowLeft': name = 'left'; break;
        case 'ArrowRight': name = 'right'; break;
        case 'Home': name = 'home'; break;
        case 'End': name = 'end'; break;
        case 'PageUp': name = 'pgup'; break;
        case 'PageDown': name = 'pgdn'; break;
        case 'Delete': name = 'delete'; break;
        default: break;
      }
      if (name) {
        event.preventDefault();
        var parts = name.split('+');
        var base = parts.pop();
        if (event.ctrlKey && parts.indexOf('ctrl') === -1) parts.push('ctrl');
        if (event.altKey && parts.indexOf('alt') === -1) parts.push('alt');
        if (event.shiftKey && parts.indexOf('shift') === -1
            && name !== 'shift+tab') parts.push('shift');
        send({ t: 'key', k: canonical(parts, base) });
        return;
      }
      if ((event.ctrlKey || event.altKey) && event.key.length === 1) {
        event.preventDefault();
        if (event.key === 'v' || event.key === 'V') {
          /* Paste what this device copied; the app's own clipboard belongs to
             the machine running AdCode, which is not where the user is. */
          pasteFromClipboard(function () { send({ t: 'key', k: 'ctrl+v' }); });
          return;
        }
        var chord = event.key.toLowerCase();
        var list = [];
        if (event.ctrlKey) list.push('ctrl');
        if (event.altKey) list.push('alt');
        if (event.shiftKey) list.push('shift');
        send({ t: 'key', k: canonical(list, chord) });
      }
    });
    tapEl.addEventListener('keyup', function (event) {
      if (event.key === 'Alt') setAltHeld(false);
    });
    window.addEventListener('blur', function () { setAltHeld(false); });
  }

  /* -- paste, scroll and taps ----------------------------------- */

  function bindGestures() {
    /* A held Alt on a real keyboard drives the same reveal as the Alt button. */
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Alt' && !event.repeat) sendModifier('alt', 'press');
    });
    document.addEventListener('keyup', function (event) {
      if (event.key === 'Alt') sendModifier('alt', 'release');
    });

    document.addEventListener('paste', function (event) {
      var data = event.clipboardData;
      if (!data) return;
      var text = data.getData('text');
      if (!text) return;
      event.preventDefault();
      sendPaste(text);
    });

    screenEl.addEventListener('wheel', function (event) {
      if (!event.deltaY) return;
      event.preventDefault();
      sendWheel(event.deltaY > 0 ? 'wheel_down' : 'wheel_up');
    }, { passive: false });

    screenEl.addEventListener('touchstart', function (event) {
      if (event.touches.length !== 1) return;
      touch.active = true;
      touch.moved = false;
      touch.y = event.touches[0].clientY;
      touch.scroll = 0;
    }, { passive: true });

    screenEl.addEventListener('touchmove', function (event) {
      if (!touch.active || event.touches.length !== 1) return;
      var y = event.touches[0].clientY;
      touch.scroll += y - touch.y;
      touch.y = y;
      if (Math.abs(touch.scroll) > 8) touch.moved = true;
      if (Math.abs(touch.scroll) >= WHEEL_STEP) {
        var notches = Math.trunc(touch.scroll / WHEEL_STEP);
        touch.scroll -= notches * WHEEL_STEP;
        var action = notches > 0 ? 'wheel_up' : 'wheel_down';
        for (var i = 0; i < Math.abs(notches); i++) sendWheel(action);
      }
      if (touch.moved) event.preventDefault();
    }, { passive: false });

    screenEl.addEventListener('touchend', function () {
      if (touch.active && !touch.moved) focusTap();
      touch.active = false;
    });

    screenEl.addEventListener('click', function (event) {
      /* A click on the terminal gives the hidden input the focus, so a
         hardware keyboard types into it.  A click that ended a text selection
         is a copy gesture instead, so it leaves the focus alone. */
      var selection = window.getSelection ? String(window.getSelection()) : '';
      if (selection.length) return;
      focusTap(event);
    });

    var timer = null;
    function scheduleRefit() {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(function () {
        timer = null;
        fit();
        sendSize();
      }, 200);
    }
    window.addEventListener('resize', scheduleRefit);
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', scheduleRefit);
    }
    window.addEventListener('orientationchange', scheduleRefit);
  }

  /* -- boot ------------------------------------------------------ */

  measureRatio();
  buildKeybar();
  bindSoftKeyboard();
  bindGestures();
  showBanner('connecting to AdCode...');
  connect();
})();

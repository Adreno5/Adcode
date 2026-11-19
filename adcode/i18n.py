"""The interface language: English and Simplified Chinese.

Every user-facing string is written in English in the source and translated
here.  The translation table is keyed by the English text itself, so a string
that has no Chinese counterpart simply stays English -- a missing translation
never breaks the interface, it only leaves it in the original language.

The active language lives in the configuration (``ui.locale``) and is read
through :func:`current_locale`, so changing it in ``/settings`` takes effect
at once.  :func:`set_locale` overrides the configured language for the
process, which is what the tests and the command line use.
"""

from __future__ import annotations

import threading
from typing import Any

#: The languages the interface ships with.
LOCALES = ('en', 'zh')
#: The language used when nothing else is configured.
DEFAULT_LOCALE = 'en'

#: The display names of the languages, in the language itself.
LOCALE_LABELS = {
    'en': 'English',
    'zh': '中文',
}

#: English -> Simplified Chinese.  Keys are the exact English strings the
#: interface uses; a key that is missing leaves the English text in place.
ZH: dict[str, str] = {
    # ── appearance ────────────────────────────────────────────
    'Appearance': '外观',
    'Theme': '主题',
    'Dark': '深色',
    'Light': '浅色',
    'Dark canvas, light grey content': '深色画布，浅灰内容',
    'Light canvas, dark content': '浅色画布，深色内容',
    'Accent colour': '强调色',
    'Custom accent (hex)': '自定义强调色（十六进制）',
    'Type a #rrggbb value': '输入 #rrggbb 格式的值',
    'Show header': '显示标题栏',
    'Show token status': '显示令牌状态',
    'Mouse wheel lines': '鼠标滚轮行数',
    'Frames per second': '每秒帧数',
    'Amber': '琥珀',
    'Rose': '玫瑰',
    'Sage': '鼠尾草绿',
    'Sand': '沙色',
    'Slate blue': '石板蓝',
    'Teal': '青绿',
    'Violet': '紫罗兰',
    'Mono': '单色',

    # ── work steps ────────────────────────────────────────────
    'Work steps': '工作步骤',
    'Collapse after completion': '完成后折叠',
    "Once the answer is in, fold the turn's steps into one line: "
    'Thought for N tokens, bash x2, edit':
        '回答完成后，将本轮步骤折叠为一行：思考 N tokens、bash x2、edit',
    'Preset': '预设',
    'Concise': '精简',
    'Detailed': '详细',
    'Standard': '标准',
    'Custom': '自定义',
    'Tools in one row, reasoning as a token count': '工具合并为一行，思考显示为令牌数',
    'Every step on its own': '每个步骤单独一行',
    'Every step, fully expanded': '每个步骤完整展开',
    'Edit the thinking and tool settings yourself': '自行调整思考与工具的显示设置',
    'Anything but Custom keeps the thinking and tool settings read-only':
        '除“自定义”外，思考与工具设置均为只读',
    'Ongoing': '进行中',
    'After completion': '完成后',
    'What is left of the reasoning once it is over': '思考结束后保留的内容',
    'Hidden': '隐藏',
    'Label only': '仅标签',
    'Partial window': '局部窗口',
    'Full': '完整',
    'Accept reasoning silently': '静默接受思考内容',
    'One animated label line': '一行带动画的标签',
    'Label text': '标签文字',
    'e.g. Thinking, Reasoning, Polishing': '例如：思考、推理、打磨',
    'Suffix animation': '后缀动画',
    'Fixed suffix': '固定后缀',
    'Rotating frames': '轮换帧',
    'Frames (comma separated)': '帧（英文逗号分隔）',
    'Frame durations (ms)': '每帧时长（毫秒）',
    'Token count': '令牌计数',
    'Append the size of the reasoning to the label': '在标签后附上思考内容的规模',
    'Drop it once the thought is over': '思考结束后不再显示',
    'One label line': '一行标签',
    'The same window the ongoing block uses': '与进行中区块相同的窗口',
    'Show everything': '显示全部',
    'Show a configurable window': '显示可配置的窗口',
    'Display style': '显示样式',
    'Normal': '普通',
    'Markdown text at full opacity': '完全不透明的 Markdown 文本',
    'Gradient fade': '渐隐',
    'Fade the top rows': '顶部行渐隐',
    'Top row alpha': '顶行透明度',
    '0 = invisible top row, 255 = no fade': '0 = 顶行不可见，255 = 不渐隐',
    'Omitted content': '省略内容',
    'No marker': '不标记',
    'Ellipsis label': '省略号标签',
    'Show a small label': '显示一个小标签',
    'Character count': '字符数',
    'Show how many characters were omitted': '显示省略了多少字符',
    'Truncate silently': '静默截断',
    'Window start (negative)': '窗口起点（负数）',
    'e.g. -5 starts five visual rows before the end': '例如 -5 表示从结尾前五行开始',
    'Window end (negative)': '窗口终点（负数）',
    'e.g. -1 includes the last visual row': '例如 -1 表示包含最后一行',

    # ── tools ─────────────────────────────────────────────────
    'Tool display': '工具显示',
    'Call line only': '仅调用行',
    'Result only': '仅结果',
    'Call line and result': '调用行与结果',
    'Both sections': '两个部分',
    'One compact call line': '一行紧凑的调用信息',
    'Only the result section': '只显示结果部分',
    'Show arguments': '显示参数',
    'Result display': '结果显示',
    'Show the complete result': '显示完整结果',
    'Maximum lines (0 = unlimited)': '最大行数（0 = 不限制）',
    'Show file diffs': '显示文件差异',
    'Chain of calls': '调用链',
    'One per row': '每行一个',
    'Every call gets its own row': '每次调用独占一行',
    'Merged': '合并',
    'A chain of calls folds into one row with the tool names and their counts':
        '连续调用折叠为一行，显示工具名与次数',

    # ── footer ────────────────────────────────────────────────
    'Footer status': '底部状态栏',
    'Running light': '运行光带',
    'Light period (s)': '光带周期（秒）',
    'Light width (cells)': '光带宽度（单元格）',
    'Light follows tok/s': '光带跟随 tok/s',
    'Reference tok/s': '参考 tok/s',
    'Slow factor without a rate': '无速率时的减速倍数',
    'Tokens per cycle': '每轮令牌数',
    'Breathing period (s)': '呼吸周期（秒）',
    'Progress bar': '进度条',
    'Sample freshness (s)': '采样有效期（秒）',
    'Samples before the bar': '显示进度条前的采样数',
    'Green when finished': '完成时显示绿色',
    'Wipe duration (s)': '擦除时长（秒）',
    'Stay green until input': '保持绿色直到输入',
    'Green colour (#rrggbb)': '绿色（#rrggbb）',
    'Empty uses the palette green': '留空则使用调色板的绿色',
    'Font colour (#rrggbb)': '文字颜色（#rrggbb）',
    'Empty uses white': '留空则使用白色',

    # ── hold / failover ───────────────────────────────────────
    'Hold': '重试保持',
    'Keep retrying': '持续重试',
    'Keep retrying a request that was cut short': '持续重试被中断的请求',
    'Maximum retries': '最大重试次数',
    'Seconds between retries': '重试间隔（秒）',
    'Failover': '故障转移',
    'Try other providers when one is down': '某个服务商故障时改用其他服务商',

    # ── reasoning / context ───────────────────────────────────
    'Reasoning': '推理',
    'Reasoning effort': '推理强度',
    'Force reasoning parameters': '强制发送推理参数',
    'Send reasoning fields even when model metadata is missing':
        '即使缺少模型元数据也发送推理字段',
    'Max output tokens': '最大输出令牌',
    'Context': '上下文',
    'Auto compact': '自动压缩',
    'Compact threshold %': '压缩阈值（%）',
    'Keep recent turns': '保留最近轮数',
    'Reserved output tokens': '预留输出令牌',
    'Compaction effort': '压缩推理强度',
    'Low': '低',
    'Medium': '中',
    'High': '高',
    'Xhigh': '超高',
    'Max': '最高',
    'Ultra': '极致',

    # ── agents / skills / security / mcp ──────────────────────
    'Sub-agents and teams': '子智能体与团队',
    'Enable sub-agents': '启用子智能体',
    'Enable agent teams': '启用智能体团队',
    'Max concurrent sub-agents': '最大并发子智能体数',
    'Max delegation depth': '最大委派深度',
    'Wait timeout (ms)': '等待超时（毫秒）',
    'Default fork turns': '默认分叉轮数',
    'All completed turns': '全部已完成的轮次',
    'No history': '不带历史',
    'Show agent activity': '显示智能体活动',
    'Show agent output': '显示智能体输出',
    'Default agent type': '默认智能体类型',
    'Agent system suffix': '智能体系统提示后缀',
    'Configured types': '已配置的类型',
    'Configured teams': '已配置的团队',
    'Skills': '技能',
    'Enable skills': '启用技能',
    'Include project skills': '包含项目技能',
    'Extra skill paths': '额外技能路径',
    'Separate paths with ;': '用 ; 分隔多个路径',
    'Scanned roots': '扫描目录',
    'Security': '安全',
    'Approval mode': '审批模式',
    'Ask': '询问',
    'Off': '关闭',
    'Ask before the shell, the network or spawning a worker':
        '执行 shell、访问网络或启动子进程前先询问',
    'Everything runs; every call is audited': '全部执行，每次调用都会记录审计',
    'Shell sandbox': 'Shell 沙箱',
    'None': '无',
    'Workspace': '工作区',
    'The shell behaves as it always has': 'shell 的行为与以往相同',
    'The shell may only write inside the project': 'shell 只能写入项目目录内',
    # ── project access ────────────────────────────────────────
    'Project access': '项目权限',
    'Project': '项目',
    'Read-only': '只读',
    'Full access': '完全访问',
    'This project': '当前项目',
    'Use the default': '跟随默认',
    'The default for every project': '所有项目的默认权限',
    'The rule for the project of this session': '本次会话所在项目的权限',
    'Follow the default level': '跟随默认权限',
    'The project can be read; nothing can be changed': '项目可读，但不可修改',
    'Files inside the project may be changed': '可以修改项目目录内的文件',
    'No boundary; the agent runs with your own rights': '没有边界，智能体与您权限相同',
    'Project access: {level}': '项目权限：{level}',
    'What the agent may do to this project': '智能体可以怎样改动这个项目',
    'Read the project; change nothing': '读取项目，不做任何修改',
    'Change files inside the project': '修改项目目录内的文件',
    'No boundary at all': '没有任何边界',
    'Unknown level: {wanted}. Use readonly, workspace or full.':
        '未知的权限等级：{wanted}。请用 readonly、workspace 或 full。',
    'Read-only: the project can be read, nothing can be changed':
        '只读：项目可以读取，但不能改动',
    'Workspace: files inside the project may be changed':
        '工作区：可以修改项目目录内的文件',
    'Full access: no boundary, the agent runs as you do':
        '完全访问：没有边界，智能体与您权限相同',
    'Allowed hosts': '允许的主机',
    'Empty allows every host': '留空表示允许所有主机',
    'Refused tools': '禁用的工具',
    'Separate names with ;': '用 ; 分隔多个名称',
    'Approval timeout (s)': '审批超时（秒）',
    'Audit trail': '审计日志',
    'The trail on disk': '磁盘上的记录',
    'MCP': 'MCP',
    'Configured servers': '已配置的服务',

    # ── language ──────────────────────────────────────────────
    'Language': '语言',
    'Interface language': '界面语言',
    'English': 'English',
    '中文': '中文',
    '简体中文': '简体中文',

    # ── recents ───────────────────────────────────────────────
    'Recents': '常用设置',
    'Frequently used settings, pinned to the top': '常用设置，置顶显示',
    'Frequently used settings': '常用设置',

    # ── commands ──────────────────────────────────────────────
    'Show commands and key bindings': '显示命令与快捷键',
    'Attach an image file or URL to your next message': '为下一条消息附加图片文件或链接',
    'Clear the conversation history': '清空对话历史',
    'Start a new conversation': '开始新的对话',
    'Open the settings browser': '打开设置页面',
    'Configure key bindings': '配置快捷键',
    'Exit AdCode': '退出 AdCode',
    'Switch model or edit capabilities': '切换模型或编辑能力',
    'Choose a model': '选择模型',
    'Edit the selected model capabilities': '编辑所选模型的能力',
    'Manage API providers': '管理 API 服务商',
    'Add a provider and fetch its models': '添加服务商并获取其模型列表',
    'Edit a provider and its models': '编辑服务商及其模型',
    'Remove a provider': '删除服务商',
    'List configured providers': '列出已配置的服务商',
    'Set the reasoning effort': '设置推理强度',
    'Set effort to low': '将推理强度设为低',
    'Set effort to medium': '将推理强度设为中',
    'Set effort to high': '将推理强度设为高',
    'Set effort to xhigh': '将推理强度设为超高',
    'Set effort to max': '将推理强度设为最高',
    'Set effort to ultra': '将推理强度设为极致',
    'Set the thinking display mode': '设置思考内容的显示方式',
    'Thinking display: hidden': '思考显示：隐藏',
    'Thinking display: label': '思考显示：仅标签',
    'Thinking display: partial': '思考显示：局部窗口',
    'Thinking display: full': '思考显示：完整',
    'Set how much of the work chain is shown': '设置工作链的展示粒度',
    'Set the maximum output tokens': '设置最大输出令牌',
    'Compact older turns into a summary': '将较早的轮次压缩为摘要',
    'Show context token usage': '显示上下文令牌用量',
    'Show the last provider usage record': '显示最近一次服务商用量记录',
    'Show the stable prompt prefix identity': '显示稳定提示前缀的标识',
    'Show the tool-call audit trail': '显示工具调用审计日志',
    'Manage durable goals': '管理长期目标',
    'Start a goal': '开始一个目标',
    'Show the active goal': '显示当前目标',
    'List goals': '列出目标',
    'Complete the active goal': '完成当前目标',
    'Pause the active goal': '暂停当前目标',
    'Resume a paused goal': '恢复已暂停的目标',
    'Cancel the active goal': '取消当前目标',
    'Mark the active goal blocked': '将当前目标标记为受阻',
    'Inspect and configure sub-agents': '查看并配置子智能体',
    'Show the agent tree': '显示智能体树',
    'Wait for agent activity': '等待智能体活动',
    'Interrupt an agent': '中断一个智能体',
    "Read one agent's output": '查看某个智能体的输出',
    'List agent types': '列出智能体类型',
    'List agent teams': '列出智能体团队',
    'Set the concurrent limit': '设置并发上限',
    'Set the delegation depth': '设置委派深度',
    'Manage MCP servers': '管理 MCP 服务',
    'Add an MCP server': '添加 MCP 服务',
    'Edit an MCP server': '编辑 MCP 服务',
    'Remove an MCP server': '删除 MCP 服务',
    'List MCP servers': '列出 MCP 服务',
    'Show MCP diagnostics': '查看 MCP 诊断信息',
    'Re-scan MCP tools': '重新扫描 MCP 工具',
    'List and reload installed skills': '列出并重新加载已安装的技能',
    'List installed skills': '列出已安装的技能',
    'Rescan the skill directories': '重新扫描技能目录',
    'Browse and load saved sessions': '浏览并加载已保存的会话',
    'Archive this session: read-only until /respawn': '归档当前会话：只读，直到 /respawn',
    'Restore this archived session': '恢复已归档的会话',
    'Delete this session for good': '彻底删除当前会话',
    'Save the current session': '保存当前会话',
    'Load a saved session': '加载已保存的会话',
    'Roll the conversation back to an earlier turn': '将对话回退到更早的一轮',
    'Rewind to a turn and edit its message': '回退到某一轮并编辑其消息',
    'Fork the conversation into a new session': '将对话分叉为新会话',
    'Ask an aside question without touching the task': '在不打断任务的前提下提问',
    'Show the Git working-tree diff': '显示 Git 工作区差异',
    'Change the canvas theme or accent colour': '更改画布主题或强调色',
    'Rename the current session': '重命名当前会话',

    # ── notices and dialogs ───────────────────────────────────
    'Settings': '设置',
    'Help': '帮助',
    'Providers': '服务商',
    'MCP servers': 'MCP 服务',
    'Thinking display': '思考显示',
    'Select model': '选择模型',
    'Save session': '保存会话',
    'Rename session': '重命名会话',
    'Delete session': '删除会话',
    'No provider configured.': '未配置服务商。',
    'No active session.': '没有活动会话。',
    'No session to delete.': '没有可删除的会话。',
    'No models configured. Use /provider add.': '未配置模型，请使用 /provider add。',
    'No session named': '没有名为',
    'was found.': '的会话。',
    'Session renamed to': '会话已重命名为',
    'Nothing to rename yet.': '暂时没有可重命名的会话。',
    'Use': '用法',
    'Rename': '重命名',
    'Session name': '会话名称',
    'This session is archived · /respawn restores it, /new starts another, '
    '/perish deletes it':
        '当前会话已归档 · /respawn 可恢复，/new 可新建，/perish 可删除',
    'archived · /respawn restores it, /new starts another':
        '已归档 · /respawn 可恢复，/new 可新建',

    # ── key bindings ─────────────────────────────────────────
    'General': '通用',
    'Clipboard': '剪贴板',
    'Editing': '编辑',
    'Navigation': '导航',
    'Quit': '退出',
    'exit AdCode immediately, whatever is on screen': '立即退出 AdCode，无论屏幕上有什么',
    'Command palette': '命令面板',
    'open the / command palette': '打开 / 命令面板',
    'open the settings browser': '打开设置页面',
    'Sessions': '会话',
    'browse and load saved sessions': '浏览并加载已保存的会话',
    'Agents': '智能体',
    'pick a sub-agent and read its output': '选择子智能体并查看其输出',
    'Copy, clear or exit': '复制、清空或退出',
    'copy selection, clear the input, or press twice to exit':
        '复制选中内容、清空输入，或连按两次退出',
    'Exit on EOF': '收到 EOF 时退出',
    'exit when the input is empty': '输入为空时退出',
    'Redraw': '重绘',
    'force a full repaint': '强制完整重绘',
    'Paste': '粘贴',
    'paste text, or attach a copied screenshot': '粘贴文本，或附加已复制的截图',
    'Attach clipboard image': '附加剪贴板图片',
    'attach the clipboard image to the message': '把剪贴板图片附加到消息中',
    'Copy last reply': '复制上一条回复',
    'copy the latest assistant reply as Markdown': '以 Markdown 复制最近的助手回复',
    'Cut': '剪切',
    'cut the selected input text': '剪切输入框中选中的文本',
    'Select all': '全选',
    'select the whole input': '选中整个输入内容',
    'Undo': '撤销',
    'undo the last input edit': '撤销上一次输入编辑',
    'Redo': '重做',
    'redo an undone input edit': '重做已撤销的输入编辑',
    'Newline': '换行',
    'insert a newline in the input': '在输入框中插入换行',
    'Complete': '补全',
    'complete the selected command': '补全所选命令',
    'Delete backward': '向前删除',
    'delete the character before the cursor': '删除光标前的字符',
    'Delete forward': '向后删除',
    'delete the character at the cursor': '删除光标处的字符',
    'Scroll up': '向上滚动',
    'scroll the transcript up a page': '向上滚动一屏对话记录',
    'Scroll down': '向下滚动',
    'scroll the transcript down a page': '向下滚动一屏对话记录',
    'History up': '历史向上',
    'previous history entry, or completion up': '上一条历史记录，或补全列表上移',
    'History down': '历史向下',
    'next history entry, or completion down': '下一条历史记录，或补全列表下移',
    'Cursor left': '光标左移',
    'move or select left': '向左移动或选择',
    'Cursor right': '光标右移',
    'move or select right': '向右移动或选择',
    'Line start': '行首',
    'move or select to the line start': '移动或选择到行首',
    'Line end': '行尾',
    'move or select to the line end': '移动或选择到行尾',
    'Input start': '输入开头',
    'jump to the start of the input': '跳到输入内容的开头',
    'Input end': '输入结尾',
    'jump to the end of the input': '跳到输入内容的结尾',

    # ── templates ────────────────────────────────────────────
    'try ': '试试 ',
    'enter / esc close': '回车 / esc 关闭',
    '↑↓ / wheel scroll · ': '↑↓ / 滚轮滚动 · ',
    'esc close': 'esc 关闭',
    '(nothing to select)': '（没有可选项）',
    'filter: ': '筛选：',
    ' · type to filter': ' · 输入以筛选',
    '↑↓ move · enter select · esc cancel': '↑↓ 移动 · 回车选择 · esc 取消',
    '↑↓ move · space toggle · enter confirm · esc cancel':
        '↑↓ 移动 · 空格切换 · 回车确认 · esc 取消',
    'enter confirm · esc cancel': '回车确认 · esc 取消',
    'enter / y confirm · esc / n cancel': '回车 / y 确认 · esc / n 取消',
    'Current name: {name}': '当前名称：{name}',
    'the clipboard is not available': '剪贴板不可用',
    'press {chord} again to exit': '再按一次 {chord} 退出',
    'copied {count} characters': '已复制 {count} 个字符',
    'no assistant reply to copy': '没有可复制的助手回复',

    # ── the status line ─────────────────────────────────────
    'ready': '就绪',
    'connecting': '连接中',
    'reconnecting': '重新连接中',
    'thinking': '思考中',
    'writing': '撰写中',
    'compacting': '压缩中',
    'aside': '旁注',
    'archived': '已归档',
    'token count': '令牌数',
    'lines': '行',
    '{count} selected · ctrl+c copies': '已选中 {count} 个字符 · ctrl+c 复制',
    'queued': '排队中',
    'image': '张图片',
    'images': '张图片',
    'aside question mode': '旁注提问模式',
    'ctrl+c again to exit': '再按一次 ctrl+c 退出',
    'agent': '个智能体',
    'agents': '个智能体',
    'steps collapse when done': '完成后折叠步骤',
    'merged steps': '合并步骤',
    '↑↓ move · enter load · del archive · alt archived · esc close':
        '↑↓ 移动 · 回车加载 · del 归档 · alt 查看已归档 · esc 关闭',
    '{count} messages': '{count} 条消息',
    'The language of the interface': '界面语言',
    'Key bindings': '快捷键',
    'press a key for «{label}» · esc cancels': '请按下一个键用于 «{label}» · esc 取消',
    'press a key · esc cancel': '请按下一个键 · esc 取消',
    'enter take over · esc cancel': '回车确认接管 · esc 取消',
    'enter rebind · r reset · esc close': '回车重新绑定 · r 重置 · esc 关闭',
    '«{key}» is bound to {other} -- take over? enter yes · esc no':
        '«{key}» 已绑定到 {other} —— 要接管吗？回车确认 · esc 取消',
    'enter              send · esc  interrupt the turn / close overlays':
        '回车              发送 · esc  中断当前轮次 / 关闭浮层',
    'mouse drag         select text in the transcript':
        '鼠标拖动          在对话记录中选择文本',
    'rebind any of these with /keys.': '可用 /keys 重新绑定以上快捷键。',
    '**Commands**': '**命令**',
    '**Keys**': '**快捷键**',
    'type a value · enter apply · esc cancel': '输入数值 · 回车应用 · esc 取消',
    '↑↓ move · enter expand/select · esc back': '↑↓ 移动 · 回车展开/选择 · esc 返回',
    '{label} is read-only under this preset; choose Custom to edit it':
        '{label} 在当前预设下为只读；切换到“自定义”后可编辑',
    'Session saved as {name}.': '会话已保存为 {name}。',
    'Session renamed to {name}.': '会话已重命名为 {name}。',
    'No session named {name} was found.': '没有找到名为 {name} 的会话。',
    'Session "{name}" is already archived.': '会话 “{name}” 已经归档。',
    'Session not found: {name}': '未找到会话：{name}',
    'Deleted "{name}".': '已删除 “{name}”。',
    '"{name}" is back · messages are accepted again.': '“{name}” 已恢复 · 可以继续发送消息。',
    'Archived "{name}" · /respawn restores it, /perish deletes it':
        '已归档 “{name}” · /respawn 可恢复，/perish 可删除',
    'Reasoning effort: {value}': '推理强度：{value}',
    'Work steps: {value}': '工作步骤：{value}',
    'Unknown command: {name} · type /help': '未知命令：{name} · 输入 /help 查看帮助',
    'Command failed: {error}': '命令执行失败：{error}',
    'Usage: {usage}': '用法：{usage}',
}

_lock = threading.RLock()
_override: str | None = None


def normalize_locale(locale: str | None) -> str:
    """The supported locale a name stands for ('' and 'zh-CN' included)."""
    name = str(locale or '').strip().lower().replace('_', '-')
    if not name:
        return DEFAULT_LOCALE
    if name in LOCALES:
        return name
    head = name.split('-', 1)[0]
    if head in LOCALES:
        return head
    return DEFAULT_LOCALE


def set_locale(locale: str | None) -> str:
    """Override the language of this process; returns the locale in force."""
    global _override
    with _lock:
        _override = None if locale is None else normalize_locale(locale)
        return _override or DEFAULT_LOCALE


def current_locale() -> str:
    """The language in force: the override, else the configured one."""
    with _lock:
        if _override is not None:
            return _override
    try:
        from adcode.config import current_config
        config = current_config()
    except Exception:
        config = None
    if config is None:
        return DEFAULT_LOCALE
    return normalize_locale(getattr(getattr(config, 'ui', None), 'locale', None))


def t(text: str, locale: str | None = None) -> str:
    """The text in the current language.

    English stays English; a Chinese locale returns the translation when
    there is one and the original text when there is not.  An explicit
    ``locale`` overrides the configured one, which is what the settings
    browser uses: it renders from its own copy of the configuration.
    """
    source = str(text)
    name = normalize_locale(locale) if locale is not None else current_locale()
    if name != 'zh':
        return source
    return ZH.get(source, source)


def tf(template: str, locale: str | None = None, **fields: Any) -> str:
    """A translated message with its placeholders filled in."""
    return t(template, locale).format(**fields)


def locale_label(locale: str) -> str:
    """The display name of a locale, in that locale's own language."""
    name = normalize_locale(locale)
    return LOCALE_LABELS.get(name, name)

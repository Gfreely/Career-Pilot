const state = {
  conversations: [],
  currentConversationId: null,
  currentConversationTitle: "未选择对话",
  currentMessages: [],
};

const CHAT_EMPTY_STATE_TEXT = "选择对话后即可查看消息";
const CHAT_PENDING_TEXT = "正在生成回答...";
const CHAT_EMPTY_ASSISTANT_TEXT = "当前未返回可展示内容。";

const dom = {
  healthStatus: document.getElementById("healthStatus"),
  conversationList: document.getElementById("conversationList"),
  conversationCount: document.getElementById("conversationCount"),
  currentConversationTitle: document.getElementById("currentConversationTitle"),
  chatMessages: document.getElementById("chatMessages"),
  chatInput: document.getElementById("chatInput"),
  chatThinking: document.getElementById("chatThinking"),
  chatModel: document.getElementById("chatModel"),
  chatPromptTemplate: document.getElementById("chatPromptTemplate"),
  sendMessageBtn: document.getElementById("sendMessageBtn"),
  scrollToLatestBtn: document.getElementById("scrollToLatestBtn"),
  profileSummary: document.getElementById("profileSummary"),
  profileFileContent: document.getElementById("profileFileContent"),
  analysisResult: document.getElementById("analysisResult"),
  interviewResult: document.getElementById("interviewResult"),
  toast: document.getElementById("toast"),
};

function showToast(message, isError = false) {
  dom.toast.textContent = message;
  dom.toast.style.background = isError ? "rgba(138, 32, 21, 0.92)" : "rgba(30, 26, 23, 0.88)";
  dom.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => dom.toast.classList.remove("show"), 2400);
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || `请求失败：${response.status}`;
    throw new Error(detail);
  }
  return data;
}

function splitTextValue(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function setActiveTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${tabName}Panel`);
  });
}

function renderConversationList() {
  dom.conversationCount.textContent = String(state.conversations.length);
  dom.conversationList.innerHTML = "";

  if (!state.conversations.length) {
    dom.conversationList.innerHTML = '<div class="empty-state">暂无对话</div>';
    return;
  }

  for (const conversation of state.conversations) {
    const button = document.createElement("button");
    button.className = "conversation-item";
    if (conversation.id === state.currentConversationId) {
      button.classList.add("active");
    }
    button.innerHTML = `
      <span class="conversation-title">${conversation.title}</span>
      <span class="conversation-time">${conversation.updated_at}</span>
    `;
    button.addEventListener("click", () => loadConversation(conversation.id));
    dom.conversationList.appendChild(button);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInlineMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  return html;
}

function renderMarkdownTable(lines) {
  const rows = lines.map((line) => line.trim());
  if (rows.length < 2) {
    return "";
  }

  const separatorPattern = /^\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$/;
  if (!separatorPattern.test(rows[1])) {
    return "";
  }

  const parseCells = (line) =>
    line
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((cell) => renderInlineMarkdown(cell.trim()));

  const headerCells = parseCells(rows[0]);
  const bodyRows = rows.slice(2).map(parseCells);

  const headerHtml = `<tr>${headerCells.map((cell) => `<th>${cell}</th>`).join("")}</tr>`;
  const bodyHtml = bodyRows
    .map((cells) => `<tr>${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("");

  return `<div class="table-wrap"><table><thead>${headerHtml}</thead><tbody>${bodyHtml}</tbody></table></div>`;
}

function renderMarkdown(text) {
  const normalized = String(text || "").replace(/\r\n/g, "\n");
  const codeBlockPattern = /```([\w-]*)\n([\s\S]*?)```/g;
  let cursor = 0;
  const htmlParts = [];

  function renderTextBlock(block) {
    const lines = block.split("\n");
    const fragments = [];
    let paragraphLines = [];
    let listType = "";
    let listItems = [];
    let quoteLines = [];

    function flushParagraph() {
      if (!paragraphLines.length) {
        return;
      }
      const paragraph = paragraphLines.map((line) => renderInlineMarkdown(line)).join("<br>");
      fragments.push(`<p>${paragraph}</p>`);
      paragraphLines = [];
    }

    function flushList() {
      if (!listItems.length) {
        return;
      }
      const tag = listType === "ol" ? "ol" : "ul";
      fragments.push(`<${tag}>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${tag}>`);
      listType = "";
      listItems = [];
    }

    function flushQuote() {
      if (!quoteLines.length) {
        return;
      }
      const quoteContent = quoteLines.map((line) => renderInlineMarkdown(line)).join("<br>");
      fragments.push(`<blockquote>${quoteContent}</blockquote>`);
      quoteLines = [];
    }

    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const trimmed = line.trim();

      const tableWindow = lines.slice(index).filter((_, offset) => offset < 12);
      const tableMatch = renderMarkdownTable(tableWindow.slice(0, 3));
      if (trimmed.includes("|") && tableMatch) {
        flushParagraph();
        flushList();
        flushQuote();

        let tableEnd = index + 2;
        while (tableEnd + 1 < lines.length && lines[tableEnd + 1].includes("|")) {
          tableEnd += 1;
        }
        fragments.push(renderMarkdownTable(lines.slice(index, tableEnd + 1)));
        index = tableEnd;
        continue;
      }

      if (!trimmed) {
        flushParagraph();
        flushList();
        flushQuote();
        continue;
      }

      const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (headingMatch) {
        flushParagraph();
        flushList();
        flushQuote();
        const level = headingMatch[1].length;
        fragments.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
        continue;
      }

      const quoteMatch = trimmed.match(/^>\s?(.*)$/);
      if (quoteMatch) {
        flushParagraph();
        flushList();
        quoteLines.push(quoteMatch[1]);
        continue;
      }

      const unorderedMatch = trimmed.match(/^[-*+]\s+(.*)$/);
      if (unorderedMatch) {
        flushParagraph();
        flushQuote();
        if (listType && listType !== "ul") {
          flushList();
        }
        listType = "ul";
        listItems.push(unorderedMatch[1]);
        continue;
      }

      const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (orderedMatch) {
        flushParagraph();
        flushQuote();
        if (listType && listType !== "ol") {
          flushList();
        }
        listType = "ol";
        listItems.push(orderedMatch[1]);
        continue;
      }

      flushList();
      flushQuote();
      paragraphLines.push(trimmed);
    }

    flushParagraph();
    flushList();
    flushQuote();

    return fragments.join("");
  }

  for (const match of normalized.matchAll(codeBlockPattern)) {
    const [fullMatch, language, code] = match;
    const preceding = normalized.slice(cursor, match.index);
    if (preceding.trim()) {
      htmlParts.push(renderTextBlock(preceding));
    }
    const codeLanguage = language ? `<span class="code-lang">${escapeHtml(language)}</span>` : "";
    htmlParts.push(
      `<pre class="code-block">${codeLanguage}<code>${escapeHtml(code.trimEnd())}</code></pre>`,
    );
    cursor = match.index + fullMatch.length;
  }

  const trailing = normalized.slice(cursor);
  if (trailing.trim()) {
    htmlParts.push(renderTextBlock(trailing));
  }

  if (!htmlParts.length) {
    return `<p>${renderInlineMarkdown(normalized)}</p>`;
  }

  return htmlParts.join("");
}

function scrollChatToBottom(behavior = "auto") {
  if (typeof dom.chatMessages.scrollTo === "function") {
    dom.chatMessages.scrollTo({ top: dom.chatMessages.scrollHeight, behavior });
    return;
  }
  scrollChatToBottom(scrollBehavior);
}

function findLastPendingAssistantMessageIndex(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.pending) {
      return index;
    }
  }
  return -1;
}

function normalizeAssistantContent(content) {
  const value = String(content ?? "");
  return value.trim() ? value : CHAT_EMPTY_ASSISTANT_TEXT;
}

function createMessageNode(message) {
  const block = document.createElement("article");
  block.className = `message ${message.role === "user" ? "user" : "assistant"}`;
  if (message.pending) {
    block.classList.add("pending");
  }

  const body = document.createElement("div");
  body.className = "message-body";

  if (message.role === "assistant") {
    body.classList.add("markdown-body");
    body.innerHTML = renderMarkdown(message.content || "");
  } else {
    body.textContent = message.content || "";
  }

  block.appendChild(body);
  return block;
}

function renderChatMessages(messages, options = {}) {
  const { scrollBehavior = "auto" } = options;
  dom.chatMessages.innerHTML = "";
  if (!messages.length) {
    dom.chatMessages.innerHTML = `<div class="empty-state">${CHAT_EMPTY_STATE_TEXT}</div>`;
    return;
  }

  for (const message of messages) {
    dom.chatMessages.appendChild(createMessageNode(message));
  }
  scrollChatToBottom(scrollBehavior);
}

function setCurrentMessages(messages, options = {}) {
  state.currentMessages = messages.map((message) => ({ ...message }));
  renderChatMessages(state.currentMessages, options);
}

function settlePendingAssistantMessage(content) {
  const nextMessages = state.currentMessages.map((message) => ({ ...message }));
  const pendingIndex = findLastPendingAssistantMessageIndex(nextMessages);
  const settledMessage = { role: "assistant", content: normalizeAssistantContent(content) };

  if (pendingIndex >= 0) {
    nextMessages[pendingIndex] = settledMessage;
  } else {
    nextMessages.push(settledMessage);
  }

  setCurrentMessages(nextMessages, { scrollBehavior: "smooth" });
}

function updatePendingAssistantMessage(content) {
  const pendingMessages = dom.chatMessages.querySelectorAll(".message.assistant.pending");
  const lastPending = pendingMessages[pendingMessages.length - 1];
  if (lastPending) {
    const body = lastPending.querySelector(".message-body");
    if (body) {
      body.innerHTML = renderMarkdown(normalizeAssistantContent(content));
    }
  }
  scrollChatToBottom("auto");
}

function discardPendingAssistantMessage() {
  setCurrentMessages(state.currentMessages.filter((message) => !message.pending));
}

async function syncConversationListInBackground() {
  try {
    await loadConversations();
  } catch (error) {
    console.warn("Conversation list refresh failed.", error);
  }
}

async function syncProfileBundleInBackground() {
  try {
    await loadProfileBundle();
  } catch (error) {
    console.warn("Profile bundle refresh failed.", error);
  }
}

async function loadHealth() {
  try {
    const data = await apiRequest("/healthz", { headers: {} });
    dom.healthStatus.textContent = `后端状态：${data.status}`;
  } catch (error) {
    dom.healthStatus.textContent = `后端状态异常：${error.message}`;
  }
}

async function loadConversations() {
  const data = await apiRequest("/api/chat/conversations", { headers: {} });
  state.conversations = data;
  renderConversationList();
}

async function createConversation() {
  const title = `前端对话 ${new Date().toLocaleString("zh-CN")}`;
  const data = await apiRequest("/api/chat/conversations", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
  await loadConversations();
  await loadConversation(data.id);
  showToast("已创建新对话");
}

async function loadConversation(conversationId) {
  const data = await apiRequest(`/api/chat/conversations/${conversationId}`, { headers: {} });
  state.currentConversationId = data.id;
  state.currentConversationTitle = data.title;
  dom.currentConversationTitle.textContent = data.title;
  renderConversationList();
  setCurrentMessages(data.messages || []);
}

async function sendMessage() {
  const message = dom.chatInput.value.trim();
  if (!message) {
    showToast("请输入消息内容", true);
    return;
  }
  if (!state.currentConversationId) {
    await createConversation();
  }

  const userMessage = { role: "user", content: message };
  const pendingAssistantMessage = { role: "assistant", content: CHAT_PENDING_TEXT, pending: true };
  const optimisticMessages = [...state.currentMessages, userMessage, pendingAssistantMessage];
  setCurrentMessages(optimisticMessages, { scrollBehavior: "smooth" });

  dom.sendMessageBtn.disabled = true;
  dom.chatThinking.textContent = "请求中...";
  try {
    const response = await fetch(`/api/chat/conversations/${state.currentConversationId}/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        model: dom.chatModel.value,
        prompt_template: dom.chatPromptTemplate.value,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `请求失败：${response.status}`);
    }

    dom.chatInput.value = "";

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let finalContent = "";
    let finalThinking = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataStr = line.slice(6).trim();
          if (!dataStr) continue;
          try {
            const data = JSON.parse(dataStr);
            if (data.thinking !== undefined) finalThinking = data.thinking;
            if (data.content !== undefined) finalContent = data.content;

            dom.chatThinking.textContent = finalThinking || "无思考过程";
            updatePendingAssistantMessage(finalContent);
          } catch (e) {
            console.warn("SSE JSON parse error:", e, "Data:", dataStr);
          }
        }
      }
    }

    settlePendingAssistantMessage(finalContent);
    showToast("消息发送成功");
    syncConversationListInBackground();
    syncProfileBundleInBackground();
  } catch (error) {
    discardPendingAssistantMessage();
    dom.chatThinking.textContent = "请求失败";
    throw error;
  } finally {
    dom.sendMessageBtn.disabled = false;
  }
}

function fillStructuredProfile(profile) {
  document.getElementById("profileMajor").value = profile.major || "";
  document.getElementById("profileDegree").value = profile.degree || "";
  document.getElementById("profileGraduationYear").value = profile.graduation_year || "";
  document.getElementById("profileExperienceLevel").value = profile.experience_level || "";
  document.getElementById("profileTargetCities").value = (profile.target_cities || []).join(", ");
  document.getElementById("profileTechStack").value = (profile.tech_stack || []).join(", ");
  document.getElementById("profileJobPreferences").value = (profile.job_preferences || []).join(", ");
  document.getElementById("profileConcerns").value = (profile.concerns || []).join(", ");
}

async function loadProfileBundle() {
  const data = await apiRequest("/api/profile", { headers: {} });
  fillStructuredProfile(data.profile);
  dom.profileSummary.textContent = data.profile_text || "暂无画像摘要";
}

async function saveStructuredProfile() {
  const payload = {
    major: document.getElementById("profileMajor").value.trim(),
    degree: document.getElementById("profileDegree").value.trim(),
    graduation_year: document.getElementById("profileGraduationYear").value.trim(),
    experience_level: document.getElementById("profileExperienceLevel").value.trim(),
    target_cities: splitTextValue(document.getElementById("profileTargetCities").value),
    tech_stack: splitTextValue(document.getElementById("profileTechStack").value),
    job_preferences: splitTextValue(document.getElementById("profileJobPreferences").value),
    concerns: splitTextValue(document.getElementById("profileConcerns").value),
  };

  const data = await apiRequest("/api/profile", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
  fillStructuredProfile(data.profile);
  dom.profileSummary.textContent = data.profile_text || "暂无画像摘要";
  showToast("结构化画像已保存");
}

async function reloadProfile() {
  const data = await apiRequest("/api/profile/reload", {
    method: "POST",
    body: JSON.stringify({}),
  });
  fillStructuredProfile(data.profile);
  dom.profileSummary.textContent = data.profile_text || "暂无画像摘要";
  showToast("画像已重新加载");
}

async function loadProfileFile() {
  const data = await apiRequest("/api/profile/file", { headers: {} });
  dom.profileFileContent.value = data.content || "";
  showToast("已读取画像文件");
}

async function saveProfileFile() {
  const data = await apiRequest("/api/profile/file", {
    method: "PUT",
    body: JSON.stringify({ content: dom.profileFileContent.value }),
  });
  fillStructuredProfile(data.profile);
  dom.profileSummary.textContent = data.profile_text || "暂无画像摘要";
  dom.profileFileContent.value = data.file.content || "";
  showToast("画像文件已保存");
}

function renderAnalysisList(title, items) {
  const values = items.length ? items.map((item) => `<li>${item}</li>`).join("") : "<li>暂无</li>";
  return `<div class="result-card"><h4>${title}</h4><ul>${values}</ul></div>`;
}

async function runAnalysis() {
  const data = await apiRequest("/api/profile/analyze", {
    method: "POST",
    body: JSON.stringify({
      model_name: document.getElementById("analysisModel").value,
      target_position: document.getElementById("analysisTargetPosition").value.trim(),
      target_city: document.getElementById("analysisTargetCity").value.trim(),
      target_direction: document.getElementById("analysisTargetDirection").value.trim(),
      notes: document.getElementById("analysisNotes").value.trim(),
      resume_content: document.getElementById("analysisResumeContent").value.trim(),
      conversation_id: state.currentConversationId,
    }),
  });

  dom.analysisResult.classList.remove("empty-state");
  dom.analysisResult.innerHTML = `
    <div class="result-card"><h4>总体判断</h4><p>${data.summary || "暂无"}</p></div>
    <div class="result-card"><h4>匹配度</h4><p>${data.match_score}</p></div>
    ${renderAnalysisList("优势", data.strengths || [])}
    ${renderAnalysisList("短板", data.gaps || [])}
    ${renderAnalysisList("风险", data.risks || [])}
    ${renderAnalysisList("行动计划", data.action_plan || [])}
    ${renderAnalysisList("推荐岗位", data.suggested_roles || [])}
    ${renderAnalysisList("面试准备重点", data.interview_focus || [])}
  `;
  showToast("分析报告生成完成");
}

async function generateInterviewQuestions() {
  const rawTypes = document.getElementById("interviewQuestionTypes").value.trim();
  const questionTypes = rawTypes ? splitTextValue(rawTypes) : ["综合问答"];
  const data = await apiRequest("/api/interview/questions/generate", {
    method: "POST",
    body: JSON.stringify({
      model_name: document.getElementById("interviewModel").value,
      target_position: document.getElementById("interviewTargetPosition").value.trim(),
      difficulty: document.getElementById("interviewDifficulty").value.trim(),
      question_count: Number(document.getElementById("interviewQuestionCount").value || 5),
      question_types: questionTypes,
      notes: document.getElementById("interviewNotes").value.trim(),
      conversation_id: state.currentConversationId,
    }),
  });

  if (!data.questions || !data.questions.length) {
    dom.interviewResult.textContent = "未生成题目";
    return;
  }

  dom.interviewResult.classList.remove("empty-state");
  dom.interviewResult.innerHTML = data.questions
    .map(
      (item, index) => `
        <article class="question-card">
          <div class="question-meta">
            <span class="pill">第 ${index + 1} 题</span>
            <span class="pill">${item.question_type || "未标注题型"}</span>
            <span class="pill">${data.difficulty}</span>
          </div>
          <h4>${item.question}</h4>
          <p><strong>考察点：</strong>${item.focus || "暂无"}</p>
          <p><strong>参考答案：</strong>${item.reference_answer || "暂无"}</p>
          <p><strong>追问：</strong>${item.follow_up || "暂无"}</p>
          <p><strong>生成原因：</strong>${item.reason || "暂无"}</p>
        </article>
      `,
    )
    .join("");
  showToast("面试题生成完成");
}

async function uploadResume() {
  const fileInput = document.getElementById("analysisResumeFile");
  if (!fileInput.files || fileInput.files.length === 0) {
    showToast("请先选择 PDF 简历文件", true);
    return;
  }
  const file = fileInput.files[0];
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showToast("只支持上传 PDF 文件", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const btn = document.getElementById("uploadResumeBtn");
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "解析中...";

  try {
    const response = await fetch("/api/profile/upload_resume", {
      method: "POST",
      body: formData,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `解析失败：${response.status}`);
    }
    document.getElementById("analysisResumeContent").value = data.markdown_content || "";
    showToast("简历解析成功");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

function bindEvents() {
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  });

  document.getElementById("newConversationBtn").addEventListener("click", async () => {
    try {
      await createConversation();
    } catch (error) {
      showToast(error.message, true);
    }
  });

  dom.scrollToLatestBtn.addEventListener("click", () => {
    scrollChatToBottom("smooth");
  });

  dom.sendMessageBtn.addEventListener("click", async () => {
    try {
      await sendMessage();
    } catch (error) {
      showToast(error.message, true);
    }
  });

  document.getElementById("saveStructuredProfileBtn").addEventListener("click", async () => {
    try {
      await saveStructuredProfile();
    } catch (error) {
      showToast(error.message, true);
    }
  });

  document.getElementById("reloadProfileBtn").addEventListener("click", async () => {
    try {
      await reloadProfile();
    } catch (error) {
      showToast(error.message, true);
    }
  });

  document.getElementById("loadProfileFileBtn").addEventListener("click", async () => {
    try {
      await loadProfileFile();
    } catch (error) {
      showToast(error.message, true);
    }
  });

  document.getElementById("saveProfileFileBtn").addEventListener("click", async () => {
    try {
      await saveProfileFile();
    } catch (error) {
      showToast(error.message, true);
    }
  });

  document.getElementById("uploadResumeBtn").addEventListener("click", uploadResume);

  document.getElementById("runAnalysisBtn").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    if (btn.disabled) return;
    btn.disabled = true;
    const originalText = btn.textContent;
    btn.textContent = "处理中...";
    try {
      await runAnalysis();
    } catch (error) {
      showToast(error.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = originalText;
    }
  });

  document.getElementById("generateInterviewBtn").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    if (btn.disabled) return;
    btn.disabled = true;
    const originalText = btn.textContent;
    btn.textContent = "生成中...";
    try {
      await generateInterviewQuestions();
    } catch (error) {
      showToast(error.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = originalText;
    }
  });
}

async function bootstrap() {
  bindEvents();
  await loadHealth();
  await loadConversations();
  await loadProfileBundle();
  await loadProfileFile();
  if (state.conversations.length) {
    await loadConversation(state.conversations[0].id);
  }
}

bootstrap().catch((error) => {
  showToast(error.message, true);
  console.error(error);
});


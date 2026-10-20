const translations = {
  en: {
    brandCaption: "LEGAL Q&A",
    checking: "Checking service",
    ready: "Service ready",
    unavailable: "Service unavailable",
    eyebrow: "EGYPTIAN CIVIL CODE",
    title: "Clarity on the law,<br />grounded in the source.",
    intro: "Ask in Arabic or English. Get a clear answer supported by relevant articles from the Egyptian Civil Code.",
    questionLabel: "YOUR QUESTION",
    questionPlaceholder: "For example: What happens if a contract is changed without agreement?",
    inputHint: "Use either language; your answer will follow your question.",
    askButton: "Ask a question",
    asking: "Finding an answer…",
    answerLabel: "GROUNDED RESPONSE",
    answerHeading: "Your answer",
    sourcesHeading: "Relevant articles",
    noSources: "No article citations were returned for this answer.",
    disclaimer: "Informational assistance only; this is not a substitute for advice from a qualified lawyer.",
    footer: "Built for exploring the Egyptian Civil Code.",
    apiDocs: "API documentation",
    emptyQuestion: "Enter a question before submitting.",
    requestFailed: "We couldn’t get an answer. Please try again.",
    networkError: "Could not connect to the service. Check that it is running and try again.",
  },
  ar: {
    brandCaption: "أسئلة قانونية وإجاباتها",
    checking: "جارٍ التحقق من الخدمة",
    ready: "الخدمة جاهزة",
    unavailable: "الخدمة غير متاحة",
    eyebrow: "القانون المدني المصري",
    title: "وضوح قانوني،<br />مستند إلى المصدر.",
    intro: "اسأل بالعربية أو الإنجليزية، واحصل على إجابة واضحة مدعومة بمواد ذات صلة من القانون المدني المصري.",
    questionLabel: "سؤالك",
    questionPlaceholder: "مثال: ما أثر تعديل العقد دون اتفاق الطرفين؟",
    inputHint: "استخدم العربية أو الإنجليزية؛ وستكون الإجابة بلغة سؤالك.",
    askButton: "اسأل سؤالًا",
    asking: "جارٍ البحث عن إجابة…",
    answerLabel: "إجابة مستندة إلى النص",
    answerHeading: "الإجابة",
    sourcesHeading: "المواد ذات الصلة",
    noSources: "لم تُذكر إحالات إلى مواد قانونية في هذه الإجابة.",
    disclaimer: "للمعلومات العامة فقط، ولا تُغني عن استشارة محامٍ مؤهل.",
    footer: "واجهة لاستكشاف القانون المدني المصري.",
    apiDocs: "توثيق الواجهة البرمجية",
    emptyQuestion: "اكتب سؤالك أولًا.",
    requestFailed: "تعذر الحصول على إجابة. يرجى المحاولة مرة أخرى.",
    networkError: "تعذر الاتصال بالخدمة. تحقق من تشغيلها ثم حاول مرة أخرى.",
  },
};

const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#ask-button");
const buttonText = askButton.querySelector("span");
const errorMessage = document.querySelector("#form-error");
const answerCard = document.querySelector("#answer-card");
const answerText = document.querySelector("#answer-text");
const sourcesBlock = document.querySelector("#sources-block");
const sourceList = document.querySelector("#source-list");
const noSources = document.querySelector("#no-sources");
const serviceStatus = document.querySelector("#service-status");
const serviceStatusText = document.querySelector("#service-status-text");
let currentLanguage = "en";

function setLanguage(language) {
  currentLanguage = translations[language] ? language : "en";
  const strings = translations[currentLanguage];
  document.documentElement.lang = currentLanguage;
  document.documentElement.dir = currentLanguage === "ar" ? "rtl" : "ltr";

  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const value = strings[element.dataset.i18n];
    if (value === undefined) return;
    if (element.id === "page-title") element.innerHTML = value;
    else element.textContent = value;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    element.placeholder = strings[element.dataset.i18nPlaceholder];
  });
  document.querySelectorAll("[data-language]").forEach((button) => {
    const active = button.dataset.language === currentLanguage;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (!askButton.disabled) buttonText.textContent = strings.askButton;
  if (serviceStatus.dataset.state) updateServiceStatus(serviceStatus.dataset.state);
}

function updateServiceStatus(state) {
  const strings = translations[currentLanguage];
  serviceStatus.dataset.state = state;
  serviceStatus.classList.toggle("is-ready", state === "ready");
  serviceStatus.classList.toggle("is-unavailable", state === "unavailable");
  serviceStatusText.textContent = strings[state] || strings.checking;
}

function displayError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearAnswer() {
  answerCard.hidden = true;
  answerText.textContent = "";
  sourceList.replaceChildren();
  sourcesBlock.hidden = true;
  noSources.hidden = true;
}

function showAnswer(data) {
  answerText.textContent = data.answer;
  sourceList.replaceChildren();
  if (data.sources.length) {
    data.sources.forEach((citation) => {
      const item = document.createElement("li");
      item.textContent = citation;
      sourceList.append(item);
    });
    sourcesBlock.hidden = false;
  } else {
    sourcesBlock.hidden = true;
    noSources.hidden = false;
  }
  answerCard.hidden = false;
}

async function checkService() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const health = await response.json();
    updateServiceStatus(response.ok && health.status === "healthy" ? "ready" : "unavailable");
  } catch {
    updateServiceStatus("unavailable");
  }
}

document.querySelectorAll("[data-language]").forEach((button) => {
  button.addEventListener("click", () => setLanguage(button.dataset.language));
});

questionInput.addEventListener("input", () => {
  questionInput.dir = /[\u0600-\u06ff]/u.test(questionInput.value) ? "rtl" : "ltr";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  clearAnswer();

  const question = questionInput.value.trim();
  if (!question) {
    displayError(translations[currentLanguage].emptyQuestion);
    questionInput.focus();
    return;
  }

  askButton.disabled = true;
  askButton.setAttribute("aria-busy", "true");
  buttonText.textContent = translations[currentLanguage].asking;
  form.classList.add("is-loading");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail;
      const serviceError = Array.isArray(detail?.errors) ? detail.errors.join("; ") : null;
      throw new Error(serviceError || (typeof detail === "string" ? detail : translations[currentLanguage].requestFailed));
    }
    showAnswer(data);
    answerCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    displayError(error instanceof TypeError ? translations[currentLanguage].networkError : error.message);
  } finally {
    askButton.disabled = false;
    askButton.removeAttribute("aria-busy");
    form.classList.remove("is-loading");
    buttonText.textContent = translations[currentLanguage].askButton;
  }
});

setLanguage("en");
checkService();

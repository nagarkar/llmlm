// The picking page's one script (llmlm.picker): tap to pick, picks in order, one Save that posts
// every choice as JSON, a saved or not-saved line, and a warning before leaving unsaved changes.
(function () {
  "use strict";
  const root = document.getElementById("picker");
  const save = document.getElementById("save");
  const status = document.getElementById("status");
  if (!root || !save || !status) return;

  const choices = Array.from(root.querySelectorAll("section.choice")).map(function (section) {
    const cards = Array.from(section.querySelectorAll(".candidate"));
    const picks = cards
      .filter(function (card) { return card.dataset.order; })
      .sort(function (a, b) { return Number(a.dataset.order) - Number(b.dataset.order); })
      .map(function (card) { return card.dataset.file; });
    return {
      id: section.dataset.choiceId,
      several: section.dataset.several === "true",
      cards: cards,
      picks: picks,
      note: section.querySelector("textarea"),
      line: section.querySelector(".picked-line"),
    };
  });

  function snapshot() {
    return JSON.stringify(choices.map(function (choice) {
      return { id: choice.id, picks: choice.picks, note: choice.note ? choice.note.value : "" };
    }));
  }
  let saved = snapshot();
  let sending = false;

  function show(state, text) {
    status.dataset.state = state;
    status.textContent = text;
  }
  function refresh() {
    if (sending) return;
    if (snapshot() !== saved) show("dirty", "Not saved yet: your changes are only on this screen.");
    else if (status.dataset.state === "dirty") show("clean", "No changes since the last save.");
  }

  function badge(choice, order) {
    if (!order) return "";
    return choice.several ? String(order) : "✓";
  }
  function draw(choice) {
    choice.cards.forEach(function (card) {
      const order = choice.picks.indexOf(card.dataset.file) + 1;
      card.dataset.order = order ? String(order) : "";
      card.querySelector(".pick").setAttribute("aria-pressed", order ? "true" : "false");
      card.querySelector(".order").textContent = badge(choice, order);
    });
    if (!choice.line) return;
    if (!choice.picks.length) choice.line.textContent = "Picked: none";
    else if (!choice.several) choice.line.textContent = "Picked: " + choice.picks[0];
    else {
      choice.line.textContent = "Picked: " + choice.picks.map(function (file, index) {
        return (index + 1) + ". " + file;
      }).join(", ");
    }
  }
  function toggle(choice, file) {
    const at = choice.picks.indexOf(file);
    if (at >= 0) choice.picks.splice(at, 1);
    else if (choice.several) choice.picks.push(file);
    else choice.picks = [file];
    draw(choice);
    refresh();
  }
  function find(card) {
    return choices.find(function (choice) { return choice.cards.indexOf(card) >= 0; });
  }

  // The full-size viewer: one dialog, built here so the page itself carries no hidden image.
  const viewer = typeof HTMLDialogElement === "function" ? buildViewer() : null;
  let shown = null; // { choice, index, opener }

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }
  function buildViewer() {
    const dialog = make("dialog", "viewer");
    dialog.setAttribute("aria-label", "Candidate at full size");
    const bar = make("div", "viewer-bar");
    const name = make("p", "viewer-name");
    name.setAttribute("aria-live", "polite");
    const previous = make("button", "button", "Previous");
    const next = make("button", "button", "Next");
    const pick = make("button", "button pick");
    pick.append(make("span", "when-off", "Pick"), make("span", "when-on", "Picked · remove"));
    const close = make("button", "button", "Close");
    [previous, next, pick, close].forEach(function (button) { button.type = "button"; });
    const image = make("img", "viewer-image");
    bar.append(name, previous, next, pick, close);
    dialog.append(bar, image);
    document.body.append(dialog);
    previous.addEventListener("click", function () { step(-1); });
    next.addEventListener("click", function () { step(1); });
    close.addEventListener("click", function () { dialog.close(); });
    pick.addEventListener("click", function () {
      toggle(shown.choice, shown.choice.cards[shown.index].dataset.file);
      fill();
    });
    dialog.addEventListener("keydown", function (event) {
      if (event.key === "ArrowLeft") step(-1);
      if (event.key === "ArrowRight") step(1);
    });
    dialog.addEventListener("close", function () {
      const opener = shown && shown.opener;
      shown = null;
      image.removeAttribute("src");
      if (opener) opener.focus();
    });
    return { dialog: dialog, name: name, image: image, pick: pick, close: close,
      previous: previous, next: next };
  }
  function fill() {
    const cards = shown.choice.cards;
    const card = cards[shown.index];
    const source = card.querySelector("img");
    viewer.image.src = card.querySelector("a.view").href;
    viewer.image.alt = source.alt;
    viewer.name.textContent = (shown.index + 1) + " of " + cards.length + ": " + card.dataset.file;
    viewer.pick.setAttribute("aria-pressed", card.dataset.order ? "true" : "false");
    viewer.pick.setAttribute("aria-label", card.querySelector(".pick").getAttribute("aria-label"));
    viewer.previous.disabled = shown.index === 0;
    viewer.next.disabled = shown.index === cards.length - 1;
  }
  function step(by) {
    const to = shown.index + by;
    if (to < 0 || to >= shown.choice.cards.length) return;
    shown.index = to;
    fill();
    if (document.activeElement && document.activeElement.disabled) viewer.close.focus();
  }

  root.addEventListener("click", function (event) {
    const button = event.target.closest("button.pick");
    if (button) {
      const card = button.closest(".candidate");
      toggle(find(card), card.dataset.file);
      return;
    }
    const link = event.target.closest("a.view");
    if (!link || !viewer) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const card = link.closest(".candidate");
    const choice = find(card);
    shown = { choice: choice, index: choice.cards.indexOf(card), opener: link };
    fill();
    viewer.dialog.showModal();
    viewer.close.focus();
  });
  root.addEventListener("input", function (event) {
    if (event.target.tagName === "TEXTAREA") refresh();
  });

  function reason(data, code) {
    if (data && Array.isArray(data.problems) && data.problems.length) return data.problems.join(" ");
    if (data && typeof data.error === "string" && data.error) return data.error;
    return "the host answered " + code + ".";
  }
  save.addEventListener("click", async function () {
    if (sending) return;
    sending = true;
    save.disabled = true;
    show("saving", "Saving…");
    const sent = snapshot();
    let outcome;
    try {
      let extra = {};
      try { extra = JSON.parse(root.dataset.postHeaders || "{}"); } catch (error) { extra = {}; }
      const answer = await fetch(root.dataset.postUrl, {
        method: "POST",
        credentials: "same-origin",
        headers: Object.assign({ "Content-Type": "application/json" }, extra),
        body: JSON.stringify({ token: root.dataset.token || "", choices: JSON.parse(sent) }),
      });
      let data = null;
      try { data = await answer.json(); } catch (error) { data = null; }
      if (!answer.ok) outcome = ["failed", "Not saved: " + reason(data, answer.status)];
      else {
        saved = sent;
        if (data && typeof data.token === "string") root.dataset.token = data.token;
        const at = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
        const more = data && typeof data.message === "string" && data.message ? " " + data.message : "";
        outcome = ["saved", "Saved at " + at + "." + more];
      }
    } catch (error) {
      outcome = ["failed", "Not saved: the host did not answer. Your changes are still on this screen."];
    }
    sending = false;
    save.disabled = false;
    show(outcome[0], outcome[1]);
    if (outcome[0] === "saved" && snapshot() !== saved) refresh();
  });

  window.addEventListener("beforeunload", function (event) {
    if (snapshot() === saved) return;
    event.preventDefault();
    event.returnValue = "";
  });
})();

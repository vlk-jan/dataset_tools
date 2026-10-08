/**
 * The scene page's notes card: compose, edit, delete. The list is server-
 * rendered; after each mutation the card and tab panel are re-fetched
 * (swap.js `refresh`), since notes also mark the windows table.
 */

import { frameField, makeButton, noteRequest, topicSelect } from "./notes_ui.js";
import { refresh } from "./swap.js";

/** Re-render the card and tab panel; reload the page if that fails. */
async function rerender() {
  try {
    await refresh(["notes", "tabpanel"]);
    const fresh = document.getElementById("notes");
    if (fresh) setup(fresh);
  } catch (error) {
    window.location.reload();
  }
}

const card = document.getElementById("notes");
if (card) setup(card);

function setup(card) {
  const scene = card.dataset.scene;
  const base = `/api/scene/${scene}/notes`;
  const nFrames = Number(card.dataset.frames) || 0;
  const form = card.querySelector("#noteform");
  const msg = card.querySelector(".notemsg");

  /** Report what the server said, in the card, rather than in a dialog. */
  function say(text, bad) {
    if (!msg) return;
    msg.textContent = text;
    msg.hidden = !text;
    msg.classList.toggle("bad", Boolean(bad));
  }

  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(form);
      const start = Number(data.get("frame_start"));
      const end = Number(data.get("frame_end"));
      const text = String(data.get("text") || "").trim();
      if (!text) return say("A note needs some text.", true);
      say("saving…");
      try {
        await noteRequest(base, "POST", {
          frames: [start, end],
          topic: data.get("topic"),
          verdict: data.get("verdict") || "",
          text,
        });
        await rerender();
      } catch (error) {
        say(`Could not save: ${error.message}`, true);
      }
    });
  }

  /** Swap a note's body for an editor over all its fields; the head stays. */
  function openEditor(item, id) {
    const paragraph = item.querySelector(".notetext");
    if (!paragraph || item.querySelector(".noteedit")) return; // already editing

    const editor = document.createElement("div");
    editor.className = "noteedit";

    const range = document.createElement("div");
    range.className = "noterow";
    const last = nFrames ? nFrames - 1 : null;
    const from = frameField(item.dataset.frameStart, "first frame of the range", last);
    const to = frameField(item.dataset.frameEnd, "last frame of the range", last);
    const topics = (card.dataset.topics || "").split(",").filter(Boolean);
    const topic = topicSelect(topics, item.dataset.topic);
    const verdicts = (card.dataset.verdicts || "").split(",").filter(Boolean);
    const verdict = topicSelect(["", ...verdicts], item.dataset.verdict || "");
    verdict.querySelector('option[value=""]').textContent = "none";
    const verdictLabel = document.createElement("label");
    verdictLabel.append("verdict ", verdict);
    const fromLabel = document.createElement("label");
    fromLabel.append("frames ", from);
    const toLabel = document.createElement("label");
    toLabel.append("to ", to);
    const topicLabel = document.createElement("label");
    topicLabel.append("topic ", topic);
    range.append(fromLabel, toLabel, topicLabel, verdictLabel);

    const box = document.createElement("textarea");
    box.rows = 3;
    box.value = paragraph.textContent.trim();
    box.setAttribute("aria-label", "note text");

    const save = makeButton("Save", "btn primary");
    const cancel = makeButton("Cancel", "btn");
    const actions = document.createElement("div");
    actions.className = "noteeditactions";
    actions.append(save, cancel);

    editor.append(range, box, actions);
    paragraph.replaceWith(editor);
    box.focus();

    cancel.addEventListener("click", () => {
      editor.replaceWith(paragraph);
      say("");
    });

    save.addEventListener("click", async () => {
      const text = box.value.trim();
      if (!text) return say("A note needs some text.", true);
      // The server orders a backwards pair.
      const start = Number(from.value);
      const end = Number(to.value);
      if (!Number.isFinite(start) || !Number.isFinite(end)) {
        return say("A note needs two frame numbers.", true);
      }
      say("saving…");
      try {
        await noteRequest(`${base}/${encodeURIComponent(id)}`, "PATCH", {
          frames: [start, end],
          topic: topic.value,
          verdict: verdict.value,
          text,
        });
        await rerender();
      } catch (error) {
        say(`Could not save: ${error.message}`, true);
      }
    });
  }

  card.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-act]");
    if (!button) return;
    const item = button.closest("li[data-note-id]");
    if (!item) return;
    const id = item.dataset.noteId;

    if (button.dataset.act === "delete") {
      if (!window.confirm("Delete this note?")) return;
      try {
        await noteRequest(`${base}/${encodeURIComponent(id)}`, "DELETE");
        await rerender();
      } catch (error) {
        say(`Could not delete: ${error.message}`, true);
      }
      return;
    }

    if (button.dataset.act === "edit") openEditor(item, id);
  });
}

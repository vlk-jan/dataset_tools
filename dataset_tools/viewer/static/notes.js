/**
 * The scene page's notes card: compose, edit, delete.
 *
 * The list itself is rendered by the server, so the notes are on the page
 * before this file loads and stay there if it fails. This only adds the three
 * mutations, and each one reloads afterwards rather than patching the DOM --
 * a note changes the windows table's markers too, and one reload is honest
 * about the whole page where hand-updating two places invites them to disagree.
 *
 * An edit reaches every field a note has -- range, topic and text -- because
 * all three are things that get typed wrong: a range pinned one segment too
 * early, a topic picked before reading the rest of the window. Correcting one
 * of those by deleting the note and writing it again would throw away its
 * author and its date, which are the parts nobody can retype.
 */

import { frameField, makeButton, noteRequest, topicSelect } from "./notes_ui.js";

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
          text,
        });
        window.location.reload();
      } catch (error) {
        say(`Could not save: ${error.message}`, true);
      }
    });
  }

  /**
   * Swap a note's rendered body for an editor over all of its fields.
   *
   * The head (range chip, topic, author, date) stays put and is left stale
   * until the reload: it is what the note *is* now, next to what it is being
   * changed to, and the reload is a moment away.
   */
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
    const fromLabel = document.createElement("label");
    fromLabel.append("frames ", from);
    const toLabel = document.createElement("label");
    toLabel.append("to ", to);
    const topicLabel = document.createElement("label");
    topicLabel.append("topic ", topic);
    range.append(fromLabel, toLabel, topicLabel);

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
      // Both ends go up together even when only one was touched: the server
      // orders a backwards pair, so typing them the other way round is not an
      // error here either.
      const start = Number(from.value);
      const end = Number(to.value);
      if (!Number.isFinite(start) || !Number.isFinite(end)) {
        return say("A note needs two frame numbers.", true);
      }
      say("saving…");
      try {
        await noteRequest(`${base}/${id}`, "PATCH", {
          frames: [start, end],
          topic: topic.value,
          text,
        });
        window.location.reload();
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
        await noteRequest(`${base}/${id}`, "DELETE");
        window.location.reload();
      } catch (error) {
        say(`Could not delete: ${error.message}`, true);
      }
      return;
    }

    if (button.dataset.act === "edit") openEditor(item, id);
  });
}

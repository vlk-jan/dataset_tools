/**
 * The scene page's notes card: compose, edit, delete. The list is server-
 * rendered; after each mutation the card and tab panel are re-fetched
 * (swap.js `refresh`), since notes also mark the windows table.
 */

import { keepOption, noteRequest } from "./notes_ui.js";
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
  const base = `/api/scene/${card.dataset.scene}/notes`;
  const msg = card.querySelector(".notemsg");

  /** Report what the server said, in the card, rather than in a dialog. */
  function say(text, bad) {
    if (!msg) return;
    msg.textContent = text;
    msg.hidden = !text;
    msg.classList.toggle("bad", Boolean(bad));
  }

  /** Show the server-rendered editor (#noteedit) under a note's head, filled. */
  function openEditor(item) {
    const paragraph = item.querySelector(".notetext");
    if (!paragraph || item.querySelector(".noteedit")) return; // already editing
    const editor = card.querySelector("#noteedit").content.firstElementChild.cloneNode(true);
    const f = editor.elements;
    f.frame_start.value = item.dataset.frameStart;
    f.frame_end.value = item.dataset.frameEnd;
    keepOption(f.topic, item.dataset.topic);
    keepOption(f.verdict, item.dataset.verdict || "");
    f.text.value = paragraph.textContent.trim();
    paragraph.hidden = true;
    paragraph.after(editor);
    f.text.focus();
  }

  // The compose form, and an editor (whose note it is under) alike.
  card.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(event.target);
    const text = String(data.get("text") || "").trim();
    if (!text) return say("A note needs some text.", true);
    const id = event.target.closest("li[data-note-id]")?.dataset.noteId;
    say("saving…");
    try {
      // The server orders a backwards pair.
      await noteRequest(id ? `${base}/${encodeURIComponent(id)}` : base, id ? "PATCH" : "POST", {
        frames: [Number(data.get("frame_start")), Number(data.get("frame_end"))],
        topic: data.get("topic"),
        verdict: data.get("verdict") || "",
        text,
      });
      await rerender();
    } catch (error) {
      say(`Could not save: ${error.message}`, true);
    }
  });

  card.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-act]");
    const item = button?.closest("li[data-note-id]");
    if (!item) return;
    const act = button.dataset.act;
    if (act === "edit") openEditor(item);
    if (act === "cancel") {
      item.querySelector(".noteedit").remove();
      item.querySelector(".notetext").hidden = false;
      say("");
    }
    if (act !== "delete" || !window.confirm("Delete this note?")) return;
    try {
      await noteRequest(`${base}/${encodeURIComponent(item.dataset.noteId)}`, "DELETE");
      await rerender();
    } catch (error) {
      say(`Could not delete: ${error.message}`, true);
    }
  });
}

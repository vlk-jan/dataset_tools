/**
 * The scene page's notes card: compose, edit, delete.
 *
 * The list itself is rendered by the server, so the notes are on the page
 * before this file loads and stay there if it fails. This only adds the three
 * mutations, and each one reloads afterwards rather than patching the DOM --
 * a note changes the windows table's markers too, and one reload is honest
 * about the whole page where hand-updating two places invites them to disagree.
 */

const card = document.getElementById("notes");
if (card) setup(card);

function setup(card) {
  const scene = card.dataset.scene;
  const base = `/api/scene/${scene}/notes`;
  const form = card.querySelector("#noteform");
  const msg = card.querySelector(".notemsg");

  /** Report what the server said, in the card, rather than in a dialog. */
  function say(text, bad) {
    if (!msg) return;
    msg.textContent = text;
    msg.hidden = !text;
    msg.classList.toggle("bad", Boolean(bad));
  }

  async function send(url, method, body) {
    const init = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) init.body = JSON.stringify(body);
    const response = await fetch(url, init);
    if (!response.ok) {
      let detail = `${response.status}`;
      try {
        const data = await response.json();
        if (data && data.detail) detail = data.detail;
      } catch (error) { /* a non-JSON error body is still a failure */ }
      throw new Error(detail);
    }
    return response.status === 204 ? null : response.json();
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
        await send(base, "POST", {
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

  card.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-act]");
    if (!button) return;
    const item = button.closest("li[data-note-id]");
    if (!item) return;
    const id = item.dataset.noteId;

    if (button.dataset.act === "delete") {
      if (!window.confirm("Delete this note?")) return;
      try {
        await send(`${base}/${id}`, "DELETE");
        window.location.reload();
      } catch (error) {
        say(`Could not delete: ${error.message}`, true);
      }
      return;
    }

    if (button.dataset.act === "edit") {
      const paragraph = item.querySelector(".notetext");
      if (item.querySelector("textarea")) return; // already editing
      const editor = document.createElement("div");
      editor.className = "noteedit";
      const box = document.createElement("textarea");
      box.rows = 3;
      box.value = paragraph.textContent.trim();
      const save = document.createElement("button");
      save.className = "btn primary";
      save.type = "button";
      save.textContent = "Save";
      const cancel = document.createElement("button");
      cancel.className = "btn";
      cancel.type = "button";
      cancel.textContent = "Cancel";
      editor.append(box, save, cancel);
      paragraph.replaceWith(editor);
      box.focus();

      cancel.addEventListener("click", () => editor.replaceWith(paragraph));
      save.addEventListener("click", async () => {
        const text = box.value.trim();
        if (!text) return say("A note needs some text.", true);
        try {
          await send(`${base}/${id}`, "PATCH", { text });
          window.location.reload();
        } catch (error) {
          say(`Could not save: ${error.message}`, true);
        }
      });
    }
  });
}

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

CLASS_CONFIG = {
    "cover": {"name": "Cover", "color": "#2563eb", "key": "1"},
    "story_page": {"name": "Story Page", "color": "#16a34a", "key": "2"},
    "annexe": {"name": "Annexe", "color": "#d97706", "key": "3"},
    "summary": {"name": "Summary", "color": "#9333ea", "key": "4"},
}
CLASS_NAMES = list(CLASS_CONFIG.keys())
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
MANIFEST_FILE = DATASET_DIR / "manifest.json"
LABELS_FILE = DATASET_DIR / "labels.json"


class DatasetValidatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Poneglyph - Page Type Dataset Validator")
        self.root.geometry("1400x900")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#0f172a")

        self.manifest = {}
        self.labels = {}
        self.volumes = []
        self.current_volume_id = None
        self.filtered_page_ids = []
        self.current_page_idx = -1
        self.tk_image = None
        self.raw_pil_image = None
        self.last_render_dims = (0, 0)
        self.resize_timer = None

        self._init_style()
        self._load_data()
        self._build_ui()
        self._bind_events()

        if self.volumes:
            self._select_volume(self.volumes[0]["id"])

        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after_idle(self.root.attributes, "-topmost", False)
        self.root.focus_force()

    def _init_style(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 10))
        self.style.configure("Treeview", background="#1e293b", foreground="#f8fafc", fieldbackground="#1e293b", rowheight=26, font=("Segoe UI", 9))
        self.style.map("Treeview", background=[("selected", "#3b82f6")], foreground=[("selected", "#ffffff")])
        self.style.configure("Treeview.Heading", background="#334155", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        self.style.configure("TCombobox", fieldbackground="#1e293b", background="#334155", foreground="#f8fafc")

    def _load_data(self):
        if not MANIFEST_FILE.exists() or not LABELS_FILE.exists():
            messagebox.showerror("Erreur", f"Fichiers introuvables dans {DATASET_DIR}")
            return
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            self.labels = json.load(f).get("labels", {})
        self.volumes = self.manifest.get("volumes", [])

    def _save_data(self):
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump({"labels": self.labels}, f, indent=2, ensure_ascii=False)

    def _build_ui(self):
        top_bar = tk.Frame(self.root, bg="#1e293b", height=54, padx=16, pady=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_bar, text="Tome :", bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.vol_combo = ttk.Combobox(top_bar, values=[f"{v['display_name']}" for v in self.volumes], state="readonly", width=18)
        self.vol_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.vol_combo.bind("<<ComboboxSelected>>", self._on_volume_combo_change)

        tk.Label(top_bar, text="Filtre :", bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.filter_var = tk.StringVar(value="Tous")
        self.filter_combo = ttk.Combobox(top_bar, textvariable=self.filter_var, values=[
            "Tous", "Non validés", "Validés", "Faible confiance (<80%)",
            "Covers", "Story Pages", "Annexes", "Summaries"
        ], state="readonly", width=16)
        self.filter_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        self.skip_validated_var = tk.BooleanVar(value=True)
        self.chk_skip = tk.Checkbutton(
            top_bar,
            text="Ignorer les pages validées",
            variable=self.skip_validated_var,
            bg="#1e293b",
            fg="#38bdf8",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#38bdf8",
            font=("Segoe UI", 9, "bold")
        )
        self.chk_skip.pack(side=tk.LEFT, padx=(0, 14))

        self.stats_label = tk.Label(top_bar, text="", bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 10, "bold"))
        self.stats_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_batch = tk.Button(top_bar, text="⚡ Valider Story >95%", bg="#047857", fg="#ffffff", activebackground="#059669", activeforeground="#ffffff", relief=tk.FLAT, padx=10, pady=3, font=("Segoe UI", 9, "bold"), cursor="hand2", command=self._batch_validate_story)
        btn_batch.pack(side=tk.RIGHT, padx=4)

        btn_save = tk.Button(top_bar, text="💾 Enregistrer", bg="#2563eb", fg="#ffffff", activebackground="#1d4ed8", activeforeground="#ffffff", relief=tk.FLAT, padx=10, pady=3, font=("Segoe UI", 9, "bold"), cursor="hand2", command=self._on_manual_save)
        btn_save.pack(side=tk.RIGHT, padx=4)

        body_frame = tk.Frame(self.root, bg="#0f172a")
        body_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(body_frame, bg="#1e293b", width=280)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        tree_scroll = ttk.Scrollbar(left_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree = ttk.Treeview(left_frame, columns=("page", "orig", "label", "stat"), show="headings", yscrollcommand=tree_scroll.set)
        tree_scroll.config(command=self.tree.yview)

        self.tree.heading("page", text="#")
        self.tree.heading("orig", text="Fichier")
        self.tree.heading("label", text="Classe")
        self.tree.heading("stat", text="État")

        self.tree.column("page", width=42, anchor=tk.CENTER)
        self.tree.column("orig", width=75, anchor=tk.W)
        self.tree.column("label", width=95, anchor=tk.W)
        self.tree.column("stat", width=42, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        right_frame = tk.Frame(body_frame, bg="#1e293b", width=320, padx=16, pady=16)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="INFORMATIONS PAGE", bg="#1e293b", fg="#64748b", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.lbl_page_info = tk.Label(right_frame, text="", bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 11, "bold"), justify=tk.LEFT)
        self.lbl_page_info.pack(anchor=tk.W, pady=(0, 16))

        tk.Label(right_frame, text="PRÉDICTION DU MODÈLE", bg="#1e293b", fg="#64748b", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.lbl_pred_badge = tk.Label(right_frame, text="", bg="#334155", fg="#ffffff", font=("Segoe UI", 11, "bold"), padx=10, pady=4)
        self.lbl_pred_badge.pack(anchor=tk.W, pady=(0, 12))

        self.prob_bars_frame = tk.Frame(right_frame, bg="#1e293b")
        self.prob_bars_frame.pack(fill=tk.X, pady=(0, 20))
        self.prob_widgets = {}
        for c_key in CLASS_NAMES:
            c_row = tk.Frame(self.prob_bars_frame, bg="#1e293b", pady=2)
            c_row.pack(fill=tk.X)
            lbl = tk.Label(c_row, text=f"{CLASS_CONFIG[c_key]['name']} :", width=11, anchor=tk.W, bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9))
            lbl.pack(side=tk.LEFT)
            bar = ttk.Progressbar(c_row, orient=tk.HORIZONTAL, length=110, mode="determinate")
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            pct_lbl = tk.Label(c_row, text="0.0%", width=6, anchor=tk.E, bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 8))
            pct_lbl.pack(side=tk.RIGHT)
            self.prob_widgets[c_key] = (bar, pct_lbl)

        tk.Label(right_frame, text="ATTRIBUER UNE CLASSE", bg="#1e293b", fg="#64748b", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 8))

        self.class_buttons = {}
        for c_key in CLASS_NAMES:
            cfg = CLASS_CONFIG[c_key]
            btn = tk.Button(
                right_frame,
                text=f"[{cfg['key']}] {cfg['name']}",
                bg=cfg["color"],
                fg="#ffffff",
                activebackground="#ffffff",
                activeforeground=cfg["color"],
                relief=tk.FLAT,
                font=("Segoe UI", 11, "bold"),
                pady=7,
                cursor="hand2",
                command=lambda k=c_key: self._set_class_and_advance(k)
            )
            btn.pack(fill=tk.X, pady=3)
            self.class_buttons[c_key] = btn

        action_frame = tk.Frame(right_frame, bg="#1e293b", pady=16)
        action_frame.pack(fill=tk.X, pady=(12, 0))

        self.btn_validate = tk.Button(
            action_frame,
            text="✔ Valider & Suivant (Espace / Entrée)",
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            relief=tk.FLAT,
            font=("Segoe UI", 10, "bold"),
            pady=8,
            cursor="hand2",
            command=self._confirm_and_advance
        )
        self.btn_validate.pack(fill=tk.X, pady=4)

        self.lbl_status_badge = tk.Label(action_frame, text="", bg="#1e293b", font=("Segoe UI", 9, "bold"))
        self.lbl_status_badge.pack(pady=4)

        shortcut_help = tk.Label(
            right_frame,
            text="Raccourcis :\n[1..4] Choisir classe & page non validée suiv.\n[Espace/Entrée] Valider & page suiv.\n[← / →] Page préc. / suiv.\n[V] Basculer validation",
            bg="#1e293b",
            fg="#64748b",
            font=("Segoe UI", 8),
            justify=tk.LEFT
        )
        shortcut_help.pack(side=tk.BOTTOM, anchor=tk.W)

        center_frame = tk.Frame(body_frame, bg="#020617")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        nav_bar = tk.Frame(center_frame, bg="#0f172a", pady=6)
        nav_bar.pack(side=tk.BOTTOM, fill=tk.X)

        btn_prev = tk.Button(nav_bar, text="◀ Précédent (←)", bg="#334155", fg="#f8fafc", activebackground="#475569", relief=tk.FLAT, padx=12, pady=4, cursor="hand2", command=self._prev_page)
        btn_prev.pack(side=tk.LEFT, padx=16)

        self.page_indicator = tk.Label(nav_bar, text="0 / 0", bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 10, "bold"))
        self.page_indicator.pack(side=tk.LEFT, expand=True)

        btn_next = tk.Button(nav_bar, text="Suivant (→) ▶", bg="#334155", fg="#f8fafc", activebackground="#475569", relief=tk.FLAT, padx=12, pady=4, cursor="hand2", command=self._next_page)
        btn_next.pack(side=tk.RIGHT, padx=16)

        self.canvas = tk.Canvas(center_frame, bg="#020617", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _bind_events(self):
        self.root.bind("<Key-1>", lambda e: self._set_class_and_advance("cover"))
        self.root.bind("<Key-2>", lambda e: self._set_class_and_advance("story_page"))
        self.root.bind("<Key-3>", lambda e: self._set_class_and_advance("annexe"))
        self.root.bind("<Key-4>", lambda e: self._set_class_and_advance("summary"))
        self.root.bind("<space>", lambda e: self._confirm_and_advance())
        self.root.bind("<Return>", lambda e: self._confirm_and_advance())
        self.root.bind("<Left>", lambda e: self._prev_page())
        self.root.bind("<Right>", lambda e: self._next_page())
        self.root.bind("<v>", lambda e: self._toggle_validation())
        self.root.bind("<V>", lambda e: self._toggle_validation())

    def _on_canvas_configure(self, event):
        if event.width < 50 or event.height < 50:
            return
        if (event.width, event.height) == self.last_render_dims:
            return
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(40, self._render_current_image)

    def _on_volume_combo_change(self, event):
        idx = self.vol_combo.current()
        if 0 <= idx < len(self.volumes):
            self._select_volume(self.volumes[idx]["id"])

    def _select_volume(self, volume_id: str):
        self.current_volume_id = volume_id
        for i, v in enumerate(self.volumes):
            if v["id"] == volume_id:
                self.vol_combo.current(i)
                break
        self._apply_filter()

    def _apply_filter(self):
        flt = self.filter_var.get()
        all_pids = [
            pid for pid, data in self.labels.items()
            if data.get("volume_id") == self.current_volume_id
        ]
        all_pids.sort(key=lambda pid: self.labels[pid].get("page_file", ""))

        filtered = []
        for pid in all_pids:
            data = self.labels[pid]
            lbl = data.get("label", "")
            stat = data.get("status", "auto_predicted")
            conf = data.get("confidence", 1.0)

            if flt == "Non validés" and stat == "validated":
                continue
            if flt == "Validés" and stat != "validated":
                continue
            if flt == "Faible confiance (<80%)" and conf >= 0.8:
                continue
            if flt == "Covers" and lbl != "cover":
                continue
            if flt == "Story Pages" and lbl != "story_page":
                continue
            if flt == "Annexes" and lbl != "annexe":
                continue
            if flt == "Summaries" and lbl != "summary":
                continue

            filtered.append(pid)

        self.filtered_page_ids = filtered
        self.current_page_idx = -1
        self._refresh_tree()
        self._update_stats()

        if self.filtered_page_ids:
            target_idx = 0
            if self.skip_validated_var.get():
                for idx, pid in enumerate(self.filtered_page_ids):
                    if self.labels[pid].get("status") != "validated":
                        target_idx = idx
                        break
            self._go_to_index(target_idx)
        else:
            self.canvas.delete("all")
            self.lbl_page_info.config(text="Aucune page ne correspond au filtre.")
            self.page_indicator.config(text="0 / 0")

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for pid in self.filtered_page_ids:
            data = self.labels[pid]
            p_file = data.get("page_num", 0)
            orig = data.get("original_file", "")
            lbl = data.get("label", "")
            stat = "✔" if data.get("status") == "validated" else "⏳"
            self.tree.insert("", tk.END, iid=pid, values=(p_file, orig, lbl, stat))

    def _update_stats(self):
        vol_pids = [pid for pid, d in self.labels.items() if d.get("volume_id") == self.current_volume_id]
        tot = len(vol_pids)
        val = sum(1 for pid in vol_pids if self.labels[pid].get("status") == "validated")
        rem = tot - val
        pct = (val / tot * 100) if tot > 0 else 0
        self.stats_label.config(text=f"Total: {tot} | Validés: {val} ({pct:.0f}%) | Restants: {rem}")

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        pid = sel[0]
        if pid in self.filtered_page_ids:
            new_idx = self.filtered_page_ids.index(pid)
            if new_idx == self.current_page_idx:
                return
            self.current_page_idx = new_idx
            self._load_current_page(pid)

    def _go_to_index(self, idx: int):
        if not self.filtered_page_ids:
            return
        new_idx = max(0, min(idx, len(self.filtered_page_ids) - 1))
        pid = self.filtered_page_ids[new_idx]
        self.current_page_idx = new_idx

        curr_sel = self.tree.selection()
        if not curr_sel or curr_sel[0] != pid:
            if self.tree.exists(pid):
                self.tree.selection_set(pid)
                self.tree.see(pid)

        self._load_current_page(pid)

    def _load_current_page(self, pid: str):
        data = self.labels[pid]
        p_num = data.get("page_num", "?")
        orig_file = data.get("original_file", "")
        self.lbl_page_info.config(text=f"Page #{p_num} : {orig_file}\n{data.get('volume_name', '')}")

        pred_lbl = data.get("predicted_label", data.get("label", ""))
        conf = data.get("confidence", 0.0)
        pred_cfg = CLASS_CONFIG.get(pred_lbl, {"name": pred_lbl, "color": "#475569"})
        self.lbl_pred_badge.config(text=f"{pred_cfg['name']} ({conf * 100:.1f}%)", bg=pred_cfg["color"])

        probs = data.get("probabilities", {})
        for c_key, (bar, pct_lbl) in self.prob_widgets.items():
            val = probs.get(c_key, 0.0) * 100
            bar["value"] = val
            pct_lbl.config(text=f"{val:.1f}%")

        curr_lbl = data.get("label", "")
        for c_key, btn in self.class_buttons.items():
            if c_key == curr_lbl:
                btn.config(relief=tk.SOLID, bd=3)
            else:
                btn.config(relief=tk.FLAT, bd=0)

        is_val = data.get("status") == "validated"
        if is_val:
            self.lbl_status_badge.config(text="✔ VALIDÉE", fg="#4ade80")
        else:
            self.lbl_status_badge.config(text="⏳ AUTO-PRÉDITE", fg="#fbbf24")

        self.page_indicator.config(text=f"{self.current_page_idx + 1} / {len(self.filtered_page_ids)}")

        img_path = DATASET_DIR / "volumes" / data["volume_id"] / "images" / data["page_file"]
        if img_path.exists():
            self.raw_pil_image = Image.open(img_path)
            self.last_render_dims = (0, 0)
            self._render_current_image()
        else:
            self.canvas.delete("all")
            self.canvas.create_text(250, 250, text=f"Image manquante :\n{img_path.name}", fill="#ef4444", font=("Segoe UI", 12))

    def _render_current_image(self):
        if not self.raw_pil_image:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        iw, ih = self.raw_pil_image.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = self.raw_pil_image.resize((nw, nh), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
        self.last_render_dims = (cw, ch)

    def _find_next_unvalidated_index(self, start_idx: int) -> int:
        n = len(self.filtered_page_ids)
        for i in range(start_idx + 1, n):
            pid = self.filtered_page_ids[i]
            if self.labels[pid].get("status") != "validated":
                return i
        for i in range(0, start_idx + 1):
            pid = self.filtered_page_ids[i]
            if self.labels[pid].get("status") != "validated":
                return i
        return -1

    def _find_prev_unvalidated_index(self, start_idx: int) -> int:
        n = len(self.filtered_page_ids)
        for i in range(start_idx - 1, -1, -1):
            pid = self.filtered_page_ids[i]
            if self.labels[pid].get("status") != "validated":
                return i
        for i in range(n - 1, start_idx - 1, -1):
            pid = self.filtered_page_ids[i]
            if self.labels[pid].get("status") != "validated":
                return i
        return -1

    def _set_class_and_advance(self, class_name: str):
        if not self.filtered_page_ids:
            return
        pid = self.filtered_page_ids[self.current_page_idx]
        self.labels[pid]["label"] = class_name
        self.labels[pid]["status"] = "validated"
        self._update_tree_row(pid)
        self._save_data()
        self._update_stats()
        self._next_page()

    def _confirm_and_advance(self):
        if not self.filtered_page_ids:
            return
        pid = self.filtered_page_ids[self.current_page_idx]
        self.labels[pid]["status"] = "validated"
        self._update_tree_row(pid)
        self._save_data()
        self._update_stats()
        self._next_page()

    def _toggle_validation(self):
        if not self.filtered_page_ids:
            return
        pid = self.filtered_page_ids[self.current_page_idx]
        curr = self.labels[pid].get("status") == "validated"
        self.labels[pid]["status"] = "auto_predicted" if curr else "validated"
        self._update_tree_row(pid)
        self._save_data()
        self._update_stats()
        self._load_current_page(pid)

    def _update_tree_row(self, pid: str):
        if self.tree.exists(pid):
            data = self.labels[pid]
            p_file = data.get("page_num", 0)
            orig = data.get("original_file", "")
            lbl = data.get("label", "")
            stat = "✔" if data.get("status") == "validated" else "⏳"
            self.tree.item(pid, values=(p_file, orig, lbl, stat))

    def _prev_page(self):
        if not self.filtered_page_ids:
            return
        if self.skip_validated_var.get():
            prev_idx = self._find_prev_unvalidated_index(self.current_page_idx)
            if prev_idx != -1 and prev_idx != self.current_page_idx:
                self._go_to_index(prev_idx)
                return
        if self.current_page_idx > 0:
            self._go_to_index(self.current_page_idx - 1)

    def _next_page(self):
        if not self.filtered_page_ids:
            return
        if self.skip_validated_var.get():
            next_idx = self._find_next_unvalidated_index(self.current_page_idx)
            if next_idx != -1 and next_idx != self.current_page_idx:
                self._go_to_index(next_idx)
                return
            else:
                all_done = all(self.labels[p].get("status") == "validated" for p in self.filtered_page_ids)
                if all_done:
                    messagebox.showinfo("Bravo !", "Toutes les pages de ce tome ont été validées !")
                    return
        if self.current_page_idx < len(self.filtered_page_ids) - 1:
            self._go_to_index(self.current_page_idx + 1)

    def _batch_validate_story(self):
        count = 0
        for pid in self.filtered_page_ids:
            data = self.labels[pid]
            if data.get("status") != "validated":
                if data.get("label") == "story_page" and data.get("confidence", 0.0) >= 0.95:
                    data["status"] = "validated"
                    count += 1
        self._save_data()
        self._refresh_tree()
        self._update_stats()
        messagebox.showinfo("Validation par lot", f"{count} pages 'story_page' avec confiance >= 95% ont été validées.")
        if self.skip_validated_var.get():
            next_idx = self._find_next_unvalidated_index(self.current_page_idx)
            if next_idx != -1:
                self._go_to_index(next_idx)

    def _on_manual_save(self):
        self._save_data()
        messagebox.showinfo("Succès", f"Labels enregistrés avec succès dans :\n{LABELS_FILE}")


def main():
    root = tk.Tk()
    app = DatasetValidatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

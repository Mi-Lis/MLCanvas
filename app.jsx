import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  MiniMap,
  useEdgesState,
  useNodesState,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Check, Code2, Download, FileUp, Group, Layers, Link as LinkIcon, Loader2, Play, Plus, Share2, Sparkles, Square, Trash2, Wand2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// --- Types ---
const NODE_TYPES = {
  Data: "Data",
  Transform: "Transform",
  Split: "Split",
  Model: "Model",
  Loss: "Loss",
  Optimizer: "Optimizer",
  Metric: "Metric",
  Trainer: "Trainer",
  Composite: "Composite",
};

const defaultViewport = { x: 0, y: 0, zoom: 0.85 };

const palette = [
  { t: NODE_TYPES.Data, color: "bg-blue-50 border-blue-200" },
  { t: NODE_TYPES.Transform, color: "bg-cyan-50 border-cyan-200" },
  { t: NODE_TYPES.Split, color: "bg-emerald-50 border-emerald-200" },
  { t: NODE_TYPES.Model, color: "bg-violet-50 border-violet-200" },
  { t: NODE_TYPES.Loss, color: "bg-rose-50 border-rose-200" },
  { t: NODE_TYPES.Optimizer, color: "bg-amber-50 border-amber-200" },
  { t: NODE_TYPES.Metric, color: "bg-slate-50 border-slate-200" },
  { t: NODE_TYPES.Trainer, color: "bg-fuchsia-50 border-fuchsia-200" },
  { t: NODE_TYPES.Composite, color: "bg-zinc-50 border-zinc-200" },
];

// --- Helpers ---
function uid(prefix = "n") {
  return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
}

function topologicalSort(nodes, edges) {
  const incoming = new Map(nodes.map((n) => [n.id, 0]));
  const out = new Map(nodes.map((n) => [n.id, []]));
  edges.forEach((e) => {
    if (!out.has(e.source)) out.set(e.source, []);
    out.get(e.source).push(e.target);
    incoming.set(e.target, (incoming.get(e.target) || 0) + 1);
  });
  const q = [...nodes.filter((n) => (incoming.get(n.id) || 0) === 0).map((n) => n.id)];
  const res = [];
  while (q.length) {
    const v = q.shift();
    res.push(v);
    (out.get(v) || []).forEach((u) => {
      incoming.set(u, (incoming.get(u) || 0) - 1);
      if ((incoming.get(u) || 0) === 0) q.push(u);
    });
  }
  return res;
}

function download(filename, text) {
  const element = document.createElement("a");
  const file = new Blob([text], { type: "text/plain" });
  element.href = URL.createObjectURL(file);
  element.download = filename;
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}

// --- Node UI ---
function NodeCard({ data, selected }) {
  const color = palette.find((p) => p.t === data.type)?.color || "bg-white border-zinc-200";
  return (
    <div className={`rounded-2xl border ${color} shadow-sm min-w-[200px] max-w-[320px]`}> 
      <div className="px-3 py-2 border-b flex items-center gap-2">
        <Badge variant="secondary" className="rounded-xl text-xs" title={data.type}>{data.type}</Badge>
        <span className="font-medium truncate" title={data.label}>{data.label}</span>
        {selected && <Sparkles className="ml-auto h-4 w-4" />}
      </div>
      <div className="p-3 text-xs text-zinc-600 whitespace-pre-wrap">
        {data.summary || "—"}
      </div>
    </div>
  );
}

const nodeTypes = { default: NodeCard };

// --- Property Editor ---
function PropertyEditor({ selection, onChange }) {
  if (!selection) return (
    <div className="text-sm text-zinc-500 p-4">Выбери узел, чтобы редактировать его параметры.</div>
  );
  const update = (patch) => onChange({ ...selection, data: { ...selection.data, ...patch } });

  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 p-4">
        <div>
          <Label>Название</Label>
          <Input value={selection.data.label || ""} onChange={(e) => update({ label: e.target.value })} />
        </div>
        <div>
          <Label>Краткое описание</Label>
          <Textarea value={selection.data.summary || ""} onChange={(e) => update({ summary: e.target.value })} rows={4} />
        </div>
        <div>
          <Label>Параметры (JSON)</Label>
          <Textarea
            value={JSON.stringify(selection.data.params || {}, null, 2)}
            onChange={(e) => {
              try {
                const val = JSON.parse(e.target.value || "{}");
                update({ params: val });
              } catch (err) {}
            }}
            rows={12}
          />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Button variant="secondary" onClick={() => update({})}><Check className="h-4 w-4 mr-1"/>Сохранить</Button>
        </div>
      </div>
    </ScrollArea>
  );
}

// --- Codegen (PyTorch training script) ---
function generatePyTorchCode(nodes, edges, projectName = "ml_pipeline") {
  const order = topologicalSort(nodes, edges);
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const orderedNodes = order.map((id) => byId[id]).filter(Boolean);

  const lines = [];
  lines.push(`# Auto-generated by ML Pipeline Visual Board`);
  lines.push(`import torch`);
  lines.push(`import torch.nn as nn`);
  lines.push(`import torch.optim as optim`);
  lines.push(`from torch.utils.data import DataLoader`);
  lines.push(`\n`);
  lines.push(`# ---- Config ----`);
  lines.push(`device = "cuda" if torch.cuda.is_available() else "cpu"`);
  lines.push(`\n`);

  // Data nodes
  const dataNodes = orderedNodes.filter((n) => n.data.type === NODE_TYPES.Data);
  if (dataNodes.length) {
    dataNodes.forEach((n, i) => {
      const name = n.data.label?.replace(/\W+/g, "_") || `dataset_${i}`;
      const params = JSON.stringify(n.data.params || {}, null, 2);
      lines.push(`# Data: ${n.data.label}`);
      lines.push(`${name} = None  # TODO: load your dataset here`);
      lines.push(`# params: ${params.split("\n").join("\n# ")}`);
    });
    lines.push(`\n`);
  }

  // Transform nodes
  const tfNodes = orderedNodes.filter((n) => n.data.type === NODE_TYPES.Transform);
  if (tfNodes.length) {
    lines.push(`# Transforms`);
    tfNodes.forEach((n, i) => {
      const name = n.data.label?.replace(/\W+/g, "_") || `transform_${i}`;
      const params = JSON.stringify(n.data.params || {}, null, 2);
      lines.push(`${name} = nn.Identity()  # TODO: implement transform`);
      lines.push(`# params: ${params.split("\n").join("\n# ")}`);
    });
    lines.push(`\n`);
  }

  // Split
  const split = orderedNodes.find((n) => n.data.type === NODE_TYPES.Split);
  if (split) {
    const p = split.data.params || { train: 0.8, val: 0.1, test: 0.1, shuffle: true };
    lines.push(`# Split`);
    lines.push(`train_ds, val_ds, test_ds = None, None, None  # TODO: split your dataset according to params`);
    lines.push(`# params: ${JSON.stringify(p, null, 2).split("\n").join("\n# ")}`);
    lines.push(`\n`);
  }

  // Model
  const model = orderedNodes.find((n) => n.data.type === NODE_TYPES.Model);
  if (model) {
    const p = model.data.params || {};
    lines.push(`# Model: ${model.data.label || "Model"}`);
    lines.push(`class GeneratedModel(nn.Module):`);
    lines.push(`    def __init__(self):`);
    lines.push(`        super().__init__()`);
    lines.push(`        # TODO: build layers based on params`);
    lines.push(`        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))`);
    lines.push(`    def forward(self, x):`);
    lines.push(`        return self.net(x)`);
    lines.push(`model = GeneratedModel().to(device)`);
    lines.push(`# params: ${JSON.stringify(p, null, 2).split("\n").join("\n# ")}`);
    lines.push(`\n`);
  }

  // Loss
  const loss = orderedNodes.find((n) => n.data.type === NODE_TYPES.Loss);
  if (loss) {
    lines.push(`# Loss`);
    lines.push(`criterion = nn.CrossEntropyLoss()`);
    lines.push(`\n`);
  }

  // Optimizer
  const opt = orderedNodes.find((n) => n.data.type === NODE_TYPES.Optimizer);
  if (opt) {
    const p = opt.data.params || { lr: 1e-3 };
    lines.push(`# Optimizer`);
    lines.push(`optimizer = optim.Adam(model.parameters(), lr=${p.lr ?? 1e-3})`);
    lines.push(`\n`);
  }

  // Trainer
  const trainer = orderedNodes.find((n) => n.data.type === NODE_TYPES.Trainer);
  if (trainer) {
    const p = trainer.data.params || { epochs: 3, batch_size: 64 };
    lines.push(`# Dataloaders`);
    lines.push(`train_loader = DataLoader(train_ds, batch_size=${p.batch_size ?? 64}, shuffle=True)  # TODO`);
    lines.push(`val_loader = DataLoader(val_ds, batch_size=${p.batch_size ?? 64})  # TODO`);
    lines.push(`\n`);
    lines.push(`# Train Loop`);
    lines.push(`for epoch in range(${p.epochs ?? 3}):`);
    lines.push(`    model.train()`);
    lines.push(`    for x, y in train_loader:`);
    lines.push(`        x, y = x.to(device), y.to(device)`);
    lines.push(`        optimizer.zero_grad()`);
    lines.push(`        out = model(x)`);
    lines.push(`        loss = criterion(out, y)`);
    lines.push(`        loss.backward()`);
    lines.push(`        optimizer.step()`);
    lines.push(`    print(f"epoch {epoch+1}: ok")`);
  }

  // Metrics
  const metricNodes = orderedNodes.filter((n) => n.data.type === NODE_TYPES.Metric);
  if (metricNodes.length) {
    lines.push(`\n# Metrics (evaluate on val_loader)`);
    lines.push(`model.eval()`);
    lines.push(`correct = 0; total = 0`);
    lines.push(`with torch.no_grad():`);
    lines.push(`    for x, y in val_loader:`);
    lines.push(`        x, y = x.to(device), y.to(device)`);
    lines.push(`        out = model(x)`);
    lines.push(`        pred = out.argmax(dim=1)`);
    lines.push(`        correct += (pred == y).sum().item()`);
    lines.push(`        total += y.size(0)`);
    lines.push(`print("val/accuracy:", correct/total if total else 0.0)`);
  }

  lines.push(`\n# Save model`);
  lines.push(`torch.save(model.state_dict(), "${projectName}.pt")`);

  return lines.join("\n");
}

// --- Main Component ---
export default function MLPipelineBoard() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selected, setSelected] = useState(null);
  const [code, setCode] = useState("");
  const [projectName, setProjectName] = useState("project");
  const flowRef = useRef(null);

  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdge({ ...params, markerEnd: { type: MarkerType.ArrowClosed } }, eds));
  }, [setEdges]);

  const addNode = (type) => {
    const id = uid("node");
    const position = { x: 100 + Math.random() * 600, y: 100 + Math.random() * 400 };
    const data = { label: `${type}`, type, params: {}, summary: "" };
    const n = { id, type: "default", position, data };
    setNodes((nds) => [...nds, n]);
    setSelected(n);
  };

  const removeSelected = () => {
    if (!selected) return;
    setEdges((eds) => eds.filter((e) => e.source !== selected.id && e.target !== selected.id));
    setNodes((nds) => nds.filter((n) => n.id !== selected.id));
    setSelected(null);
  };

  const onSelectionChange = useCallback(({ nodes }) => {
    setSelected(nodes[0] || null);
  }, []);

  const updateNode = (patched) => {
    setNodes((nds) => nds.map((n) => (n.id === patched.id ? { ...n, data: patched.data } : n)));
    setSelected(patched);
  };

  const validateGraph = () => {
    const errors = [];
    const typeCount = (t) => nodes.filter((n) => n.data.type === t).length;
    if (typeCount(NODE_TYPES.Model) === 0) errors.push("Нет узла Model");
    if (typeCount(NODE_TYPES.Loss) === 0) errors.push("Нет узла Loss");
    if (typeCount(NODE_TYPES.Optimizer) === 0) errors.push("Нет узла Optimizer");
    if (errors.length) return { ok: false, errors };

    // cycle check via topological order length
    const order = topologicalSort(nodes, edges);
    if (order.length !== nodes.length) errors.push("Обнаружен цикл в графе");

    return { ok: errors.length === 0, errors };
  };

  const groupSelection = () => {
    const sel = selected ? [selected.id] : [];
    const ids = new Set(sel);
    if (ids.size === 0) return;
    const compositeId = uid("comp");
    const composite = {
      id: compositeId,
      type: "default",
      position: { x: (selected?.position?.x || 100) + 60, y: (selected?.position?.y || 100) + 60 },
      data: { label: "Composite", type: NODE_TYPES.Composite, params: { members: [...ids] }, summary: "Группа узлов" },
    };
    setNodes((nds) => [...nds, composite]);
  };

  const exportJSON = () => {
    const payload = JSON.stringify({ nodes, edges, projectName }, null, 2);
    download(`${projectName || "project"}.json`, payload);
  };

  const importJSON = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const obj = JSON.parse(String(e.target?.result || "{}"));
        setNodes(obj.nodes || []);
        setEdges(obj.edges || []);
        setProjectName(obj.projectName || "project");
      } catch (err) {
        console.error(err);
      }
    };
    reader.readAsText(file);
  };

  const buildCode = () => {
    const v = validateGraph();
    if (!v.ok) {
      setCode(`# Ошибки графа:\n# - ${v.errors.join("\n# - ")}`);
      return;
    }
    const text = generatePyTorchCode(nodes, edges, projectName);
    setCode(text);
  };

  const downloadCode = () => {
    if (!code) buildCode();
    download(`${projectName || "project"}.py`, code || generatePyTorchCode(nodes, edges, projectName));
  };

  const onDrop = useCallback((evt) => {
    evt.preventDefault();
    if (evt.dataTransfer?.files?.length) {
      const file = evt.dataTransfer.files[0];
      if (file.name.endsWith(".json")) importJSON(file);
    }
  }, []);

  const onDragOver = useCallback((evt) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = "copy";
  }, []);

  return (
    <div className="h-[100vh] w-full grid grid-cols-12 gap-3 p-3 bg-gradient-to-br from-white to-zinc-50" onDrop={onDrop} onDragOver={onDragOver}>
      {/* Left: Toolbox */}
      <div className="col-span-2 space-y-3">
        <Card className="rounded-2xl shadow-sm">
          <CardHeader>
            <CardTitle className="text-lg">Ноды</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-2">
            {palette.map((p) => (
              <Button key={p.t} variant="secondary" className="justify-start gap-2" onClick={() => addNode(p.t)}>
                <Plus className="h-4 w-4" /> {p.t}
              </Button>
            ))}
          </CardContent>
        </Card>

        <Card className="rounded-2xl">
          <CardHeader>
            <CardTitle className="text-lg">Проект</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Label>Название</Label>
            <Input value={projectName} onChange={(e) => setProjectName(e.target.value)} />
            <div className="grid grid-cols-2 gap-2 pt-1">
              <Button onClick={exportJSON} variant="outline"><Download className="h-4 w-4 mr-1"/>JSON</Button>
              <label className="cursor-pointer inline-flex items-center justify-center rounded-md border px-3 py-2 text-sm font-medium">
                <FileUp className="h-4 w-4 mr-1"/> Импорт
                <input type="file" accept=".json" className="hidden" onChange={(e) => e.target.files?.[0] && importJSON(e.target.files[0])} />
              </label>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-2xl">
          <CardHeader>
            <CardTitle className="text-lg">Операции</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-2">
            <Button onClick={buildCode} className="justify-start gap-2"><Code2 className="h-4 w-4"/>Сгенерировать код</Button>
            <Button onClick={downloadCode} variant="secondary" className="justify-start gap-2"><Download className="h-4 w-4"/>Скачать .py</Button>
            <Button onClick={groupSelection} variant="outline" className="justify-start gap-2"><Group className="h-4 w-4"/>Группировать</Button>
            <Button onClick={removeSelected} variant="destructive" className="justify-start gap-2"><Trash2 className="h-4 w-4"/>Удалить выбранный</Button>
          </CardContent>
        </Card>
      </div>

      {/* Center: Canvas */}
      <div className="col-span-7 rounded-2xl overflow-hidden border bg-white shadow-sm">
        <ReactFlow
          ref={flowRef}
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onSelectionChange={onSelectionChange}
          fitView
          defaultViewport={defaultViewport}
        >
          <MiniMap pannable zoomable className="rounded-xl" />
          <Controls />
          <Background variant="dots" gap={18} size={1} />
        </ReactFlow>
      </div>

      {/* Right: Properties & Code */}
      <div className="col-span-3 flex flex-col gap-3">
        <Card className="rounded-2xl h-[48%]">
          <CardHeader>
            <CardTitle className="text-lg">Свойства</CardTitle>
          </CardHeader>
          <CardContent className="h-[calc(100%-4rem)]">
            <PropertyEditor selection={selected} onChange={updateNode} />
          </CardContent>
        </Card>

        <Card className="rounded-2xl h-[48%]">
          <CardHeader className="flex items-center justify-between">
            <CardTitle className="text-lg">Код (PyTorch)</CardTitle>
          </CardHeader>
          <CardContent className="h-[calc(100%-4rem)]">
            <ScrollArea className="h-full border rounded-xl p-3 bg-zinc-50 text-xs">
              <pre className="whitespace-pre-wrap leading-relaxed">{code || "# Нажми \"Сгенерировать код\""}</pre>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

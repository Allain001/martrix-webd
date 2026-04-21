import { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_MATRIX = [
  [1.2, 0.4],
  [-0.3, 1.1],
];

const DEFAULT_B = [2, 1];

const OPERATIONS = [
  { value: "all", label: "全链路分析" },
  { value: "determinant", label: "行列式" },
  { value: "inverse", label: "逆矩阵" },
  { value: "eigen", label: "特征值" },
  { value: "rank", label: "矩阵秩" },
  { value: "solve", label: "求解 Ax = b" },
];

const GRAPH_MODES = [
  { value: "split", label: "双视图" },
  { value: "source", label: "原始平面" },
  { value: "target", label: "变换平面" },
];

const INSPECTOR_TABS = [
  { value: "results", label: "结果" },
  { value: "steps", label: "步骤" },
  { value: "share", label: "分享" },
];

const OPERATIONS_SET = new Set(OPERATIONS.map((item) => item.value));

function parseMatrixParam(value) {
  if (!value) return null;
  const rows = value
    .split(";")
    .map((row) => row.split(",").map((cell) => Number(cell)));
  if (
    rows.length !== 2 ||
    rows.some(
      (row) => row.length !== 2 || row.some((cell) => Number.isNaN(cell)),
    )
  ) {
    return null;
  }
  return rows;
}

function parseVectorParam(value) {
  if (!value) return null;
  const items = value.split(",").map((cell) => Number(cell));
  if (items.length !== 2 || items.some((cell) => Number.isNaN(cell))) {
    return null;
  }
  return items;
}

function readInitialState() {
  if (typeof window === "undefined") {
    return {
      matrix: DEFAULT_MATRIX,
      bVector: DEFAULT_B,
      operation: "all",
      caseId: "",
      view: "public",
    };
  }

  const params = new URLSearchParams(window.location.search);
  const matrix = parseMatrixParam(params.get("m")) ?? DEFAULT_MATRIX;
  const bVector = parseVectorParam(params.get("b")) ?? DEFAULT_B;
  const operationParam = params.get("op");
  const operation = OPERATIONS_SET.has(operationParam) ? operationParam : "all";
  const caseId = params.get("case") ?? "";
  const view = params.get("view") === "defense" ? "defense" : "public";

  return { matrix, bVector, operation, caseId, view };
}

function formatMatrixParam(matrix) {
  return matrix
    .map((row) => row.map((value) => Number(value).toFixed(2)).join(","))
    .join(";");
}

function formatVectorParam(vector) {
  return vector.map((value) => Number(value).toFixed(2)).join(",");
}

function buildShareUrl({
  matrix,
  bVector,
  operation,
  caseId = "",
  defenseMode = false,
}) {
  const url = new URL(window.location.href);
  url.searchParams.set("m", formatMatrixParam(matrix));
  url.searchParams.set("b", formatVectorParam(bVector));
  url.searchParams.set("op", operation);
  if (caseId) {
    url.searchParams.set("case", caseId);
  } else {
    url.searchParams.delete("case");
  }
  if (defenseMode) {
    url.searchParams.set("view", "defense");
  } else {
    url.searchParams.delete("view");
  }
  return url.toString();
}

function multiplyMatrixVector(matrix, point) {
  return [
    matrix[0][0] * point[0] + matrix[0][1] * point[1],
    matrix[1][0] * point[0] + matrix[1][1] * point[1],
  ];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function roundNumber(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function solutionLabel(solution) {
  if (!solution) return "等待分析";
  return solution.type_label || solution.warning || "等待分析";
}

function geometryNarrative(determinantEstimate) {
  if (Math.abs(determinantEstimate) < 0.1) {
    return "面积接近塌缩，说明这个变换正在把平面压扁。";
  }
  if (determinantEstimate < 0) {
    return "方向发生翻转，说明这个变换包含镜像性质。";
  }
  return "方向保持不变，主要体现拉伸、旋转或剪切。";
}

function matrixFormula(matrix) {
  return `[[${matrix[0].join(", ")}], [${matrix[1].join(", ")}]]`;
}

function vectorFormula(vector) {
  return `[${vector.join(", ")}]`;
}

function App() {
  const initialState = useMemo(() => readInitialState(), []);
  const [matrix, setMatrix] = useState(initialState.matrix);
  const [bVector, setBVector] = useState(initialState.bVector);
  const [operation, setOperation] = useState(initialState.operation);
  const [activeCaseId, setActiveCaseId] = useState(initialState.caseId);
  const [defenseMode, setDefenseMode] = useState(initialState.view === "defense");
  const [graphMode, setGraphMode] = useState("split");
  const [inspectorTab, setInspectorTab] = useState("results");
  const [probePoint, setProbePoint] = useState([1.35, 0.95]);
  const [cases, setCases] = useState([]);
  const [siteContent, setSiteContent] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [shareMessage, setShareMessage] = useState("");

  const roundedMatrix = useMemo(
    () => matrix.map((row) => row.map((value) => Number(value))),
    [matrix],
  );

  const determinantEstimate = useMemo(
    () =>
      roundedMatrix[0][0] * roundedMatrix[1][1] -
      roundedMatrix[0][1] * roundedMatrix[1][0],
    [roundedMatrix],
  );

  const transformedProbe = useMemo(
    () => multiplyMatrixVector(roundedMatrix, probePoint),
    [roundedMatrix, probePoint],
  );

  const activeCase = useMemo(
    () => cases.find((item) => item.id === activeCaseId) ?? null,
    [cases, activeCaseId],
  );

  const fetchAnalysis = async (
    nextOperation = operation,
    nextMatrix = roundedMatrix,
    nextB = bVector,
  ) => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          matrix: nextMatrix,
          operation: nextOperation,
          b: nextB,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "分析请求失败");
      }
      setAnalysis(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const boot = async () => {
      try {
        const [contentResponse, casesResponse] = await Promise.all([
          fetch("/api/site-content"),
          fetch("/api/demo-cases"),
        ]);
        const contentData = await contentResponse.json();
        const casesData = await casesResponse.json();
        const fetchedCases = casesData.items || [];
        setSiteContent(contentData);
        setCases(fetchedCases);

        const matchedCase = fetchedCases.find(
          (item) => item.id === initialState.caseId,
        );
        if (matchedCase) {
          setMatrix(matchedCase.matrix);
          setBVector(matchedCase.b ?? DEFAULT_B);
          setOperation(matchedCase.operation ?? "all");
          setActiveCaseId(matchedCase.id);
          fetchAnalysis(
            matchedCase.operation ?? "all",
            matchedCase.matrix,
            matchedCase.b ?? DEFAULT_B,
          );
          return;
        }
      } catch {
        setCases([]);
      }

      fetchAnalysis(
        initialState.operation,
        initialState.matrix,
        initialState.bVector,
      );
    };

    boot();
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set("m", formatMatrixParam(matrix));
    params.set("b", formatVectorParam(bVector));
    params.set("op", operation);
    if (activeCaseId) {
      params.set("case", activeCaseId);
    } else {
      params.delete("case");
    }
    if (defenseMode) {
      params.set("view", "defense");
    } else {
      params.delete("view");
    }
    const nextUrl = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
    window.history.replaceState({}, "", nextUrl);
  }, [matrix, bVector, operation, activeCaseId, defenseMode]);

  useEffect(() => {
    if (!shareMessage) return undefined;
    const timer = window.setTimeout(() => setShareMessage(""), 3200);
    return () => window.clearTimeout(timer);
  }, [shareMessage]);

  const summaryCards = useMemo(() => {
    const results = analysis?.results ?? {};
    return [
      {
        label: "det(A)",
        value: results.determinant?.display ?? roundNumber(determinantEstimate),
      },
      {
        label: "面积缩放",
        value: `${roundNumber(determinantEstimate)}x`,
      },
      {
        label: "方程组",
        value: solutionLabel(results.solution),
      },
      {
        label: "矩阵秩",
        value:
          results.rank?.rank !== undefined
            ? `${results.rank.rank}`
            : analysis?.diagnostics?.rank_estimate ?? "2",
      },
    ];
  }, [analysis, determinantEstimate]);

  const steps = useMemo(() => {
    const results = analysis?.results;
    if (!results) return [];

    if (results.solution?.steps?.length) {
      return results.solution.steps.slice(0, 6);
    }
    if (results.inverse?.steps?.length) {
      return results.inverse.steps.slice(0, 6);
    }
    if (results.determinant?.steps?.length) {
      return results.determinant.steps.slice(0, 6);
    }
    return [];
  }, [analysis]);

  const quickFacts = useMemo(() => {
    const quickstart = siteContent?.quickstart ?? [];
    const deployment = siteContent?.deployment ?? [];
    return [...quickstart.slice(0, 2), ...deployment.slice(0, 1)];
  }, [siteContent]);

  const algebraItems = useMemo(() => {
    return [
      {
        label: "矩阵 A",
        value: matrixFormula(roundedMatrix),
      },
      {
        label: "向量 b",
        value: vectorFormula(bVector),
      },
      {
        label: "当前模式",
        value: OPERATIONS.find((item) => item.value === operation)?.label || "全链路分析",
      },
      {
        label: "当前场景",
        value: activeCase?.title || "自定义矩阵",
      },
    ];
  }, [roundedMatrix, bVector, operation, activeCase]);

  const copyPresetLink = async ({
    label,
    nextDefenseMode = false,
    nextCaseId = "",
  }) => {
    const shareUrl = buildShareUrl({
      matrix,
      bVector,
      operation,
      caseId: nextCaseId,
      defenseMode: nextDefenseMode,
    });
    try {
      await navigator.clipboard.writeText(shareUrl);
      setShareMessage(`已复制${label}`);
    } catch {
      setShareMessage(`复制失败，请手动复制这个链接：${shareUrl}`);
    }
  };

  const resetWorkspace = () => {
    setMatrix(DEFAULT_MATRIX);
    setBVector(DEFAULT_B);
    setOperation("all");
    setActiveCaseId("");
    setProbePoint([1.35, 0.95]);
    fetchAnalysis("all", DEFAULT_MATRIX, DEFAULT_B);
  };

  const diagnostics = analysis?.diagnostics ?? {};
  const results = analysis?.results ?? {};

  return (
    <div className={`app-shell ${defenseMode ? "defense-mode" : ""}`}>
      <header className="topbar">
        <div className="brand-group">
          <div className="brand-mark">M</div>
          <div>
            <div className="brand-kicker">MatrixVis Lab</div>
            <h1>MatrixVis</h1>
          </div>
        </div>

        <div className="topbar-center">
          <span>场景化矩阵变换实验室</span>
          <span>2×2 交互几何</span>
          <span>GeoGebra x Immersive Math</span>
        </div>

        <div className="topbar-actions">
          <button
            className={`mode-toggle ${defenseMode ? "active" : ""}`}
            onClick={() => setDefenseMode((current) => !current)}
          >
            {defenseMode ? "退出答辩模式" : "切换答辩模式"}
          </button>
          <button className="ghost-button" onClick={() => copyPresetLink({
            label: "当前工作台链接",
            nextDefenseMode: defenseMode,
            nextCaseId: activeCaseId,
          })}>
            复制链接
          </button>
        </div>
      </header>

      <main className="app-main">
        <section className="command-banner">
          <div>
            <span className="section-label">Scene</span>
            <strong>{activeCase?.title || "自定义矩阵实验"}</strong>
            <p>
              {activeCase?.story ||
                "拖动左侧探针点，观察单位方格、基向量和结果面板如何同步变化。"}
            </p>
          </div>
          <div className="banner-actions">
            <button
              className="primary-button"
              onClick={() => fetchAnalysis()}
              disabled={loading}
            >
              {loading ? "计算中..." : "刷新分析"}
            </button>
            <button className="ghost-button" onClick={resetWorkspace}>
              重置实验
            </button>
          </div>
        </section>

        {shareMessage && <div className="flash-message">{shareMessage}</div>}

        <section className="app-frame">
          <aside className="side-pane control-pane">
            <div className="pane-header">
              <span className="section-label">Objects</span>
              <strong>代数与场景</strong>
            </div>

            <div className="object-list">
              {algebraItems.map((item) => (
                <div className="object-row" key={item.label}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>

            <div className="pane-header compact">
              <span className="section-label">Cases</span>
              <strong>案例库</strong>
            </div>
            <div className="case-stack">
              {cases.map((item) => (
                <button
                  className={`case-button ${item.id === activeCaseId ? "active" : ""}`}
                  key={item.id}
                  onClick={() => {
                    setMatrix(item.matrix);
                    setBVector(item.b ?? DEFAULT_B);
                    setOperation(item.operation ?? "all");
                    setActiveCaseId(item.id);
                    setProbePoint([1.2, 0.8]);
                    fetchAnalysis(item.operation ?? "all", item.matrix, item.b ?? DEFAULT_B);
                  }}
                >
                  <span>{item.title}</span>
                  <strong>{item.subtitle}</strong>
                </button>
              ))}
            </div>

            <div className="pane-header compact">
              <span className="section-label">Input</span>
              <strong>矩阵控制台</strong>
            </div>

            <label className="field-block">
              <span>矩阵 A</span>
              <div className="matrix-grid">
                {matrix.map((row, rowIndex) =>
                  row.map((value, colIndex) => (
                    <input
                      key={`${rowIndex}-${colIndex}`}
                      type="number"
                      step="0.1"
                      value={value}
                      onChange={(event) => {
                        const next = matrix.map((sourceRow) => [...sourceRow]);
                        next[rowIndex][colIndex] = Number(event.target.value);
                        setMatrix(next);
                        setActiveCaseId("");
                      }}
                    />
                  )),
                )}
              </div>
            </label>

            <label className="field-block">
              <span>向量 b</span>
              <div className="vector-grid">
                {bVector.map((value, index) => (
                  <input
                    key={index}
                    type="number"
                    step="0.1"
                    value={value}
                    onChange={(event) => {
                      const next = [...bVector];
                      next[index] = Number(event.target.value);
                      setBVector(next);
                      setActiveCaseId("");
                    }}
                  />
                ))}
              </div>
            </label>

            <label className="field-block">
              <span>分析模式</span>
              <select
                value={operation}
                onChange={(event) => {
                  setOperation(event.target.value);
                  setActiveCaseId("");
                }}
              >
                {OPERATIONS.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          </aside>

          <section className="stage-pane">
            <div className="stage-topbar">
              <div>
                <span className="section-label">Stage</span>
                <strong>
                  {defenseMode ? "答辩演示视图" : "沉浸式矩阵变换舞台"}
                </strong>
              </div>

              <div className="toolbar-group">
                {GRAPH_MODES.map((mode) => (
                  <button
                    key={mode.value}
                    className={`toolbar-button ${graphMode === mode.value ? "active" : ""}`}
                    onClick={() => setGraphMode(mode.value)}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
            </div>

            <div className={`graph-stage graph-mode-${graphMode}`}>
              {graphMode === "split" ? (
                <TransformWorkspace
                  matrix={roundedMatrix}
                  probePoint={probePoint}
                  onProbeChange={setProbePoint}
                />
              ) : (
                <div className="single-graph-wrap">
                  <GraphPanel
                    title={graphMode === "source" ? "原始平面" : "变换后平面"}
                    mode={graphMode === "source" ? "source" : "target"}
                    matrix={roundedMatrix}
                    probePoint={probePoint}
                    onProbeChange={graphMode === "source" ? setProbePoint : undefined}
                    size={480}
                  />
                </div>
              )}
            </div>

            <div className="stage-readout">
              <MetricChip label="探针点">
                ({roundNumber(probePoint[0], 2)}, {roundNumber(probePoint[1], 2)})
              </MetricChip>
              <MetricChip label="变换后">
                ({roundNumber(transformedProbe[0], 2)}, {roundNumber(transformedProbe[1], 2)})
              </MetricChip>
              <MetricChip label="几何提示">
                {geometryNarrative(determinantEstimate)}
              </MetricChip>
            </div>

            <div className="hint-strip">
              <span>拖动左侧橙色点查看映射</span>
              <span>切换案例快速进入讲解情境</span>
              <span>复制答辩链接可直接进入演示视图</span>
            </div>
          </section>

          <aside className="side-pane inspector-pane">
            <div className="pane-header">
              <span className="section-label">Inspector</span>
              <strong>结果与讲解</strong>
            </div>

            <div className="tab-row">
              {INSPECTOR_TABS.map((tab) => (
                <button
                  key={tab.value}
                  className={`tab-button ${inspectorTab === tab.value ? "active" : ""}`}
                  onClick={() => setInspectorTab(tab.value)}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {error && <div className="error-box">{error}</div>}

            {inspectorTab === "results" && (
              <div className="inspector-stack">
                <div className="summary-grid">
                  {summaryCards.map((card) => (
                    <article className="summary-card" key={card.label}>
                      <span>{card.label}</span>
                      <strong>{card.value}</strong>
                    </article>
                  ))}
                </div>

                <InfoBlock title="条件数">
                  {diagnostics.condition_number
                    ? roundNumber(diagnostics.condition_number, 2)
                    : "待计算"}
                </InfoBlock>
                <InfoBlock title="特征值">
                  {results.eigen?.values?.join(" , ") || "等待分析"}
                </InfoBlock>
                <InfoBlock title="方程组结论">
                  {solutionLabel(results.solution)}
                </InfoBlock>
                <InfoBlock title="当前案例重点">
                  {activeCase?.teaching_focus || "你可以直接修改矩阵，探索自己的变换。"}
                </InfoBlock>
              </div>
            )}

            {inspectorTab === "steps" && (
              <div className="inspector-stack">
                {loading && <p className="panel-hint">正在请求新的分析结果，请稍等。</p>}
                {steps.length ? (
                  <div className="steps-list">
                    {steps.map((step, index) => (
                      <article className="step-card" key={`${step.description}-${index}`}>
                        <span>Step {index + 1}</span>
                        <strong>{step.description}</strong>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="panel-hint">
                    点击“刷新分析”后，这里会展示与当前运算同步的推导步骤。
                  </p>
                )}
              </div>
            )}

            {inspectorTab === "share" && (
              <div className="inspector-stack">
                <ShareCard
                  title="公开链接"
                  description="适合发给老师或评委自由浏览，保持当前矩阵和运算状态。"
                  onCopy={() =>
                    copyPresetLink({
                      label: "公开链接",
                      nextDefenseMode: false,
                      nextCaseId: activeCaseId,
                    })
                  }
                />
                <ShareCard
                  title="答辩链接"
                  description="直接进入更聚焦的展示视图，更适合现场投屏和讲解。"
                  onCopy={() =>
                    copyPresetLink({
                      label: "答辩链接",
                      nextDefenseMode: true,
                      nextCaseId: activeCaseId,
                    })
                  }
                />
                <ShareCard
                  title="案例直达链接"
                  description={
                    activeCase
                      ? `直接进入 ${activeCase.title} 的讲解场景。`
                      : "当前没有选中案例，会复制当前自定义矩阵状态。"
                  }
                  onCopy={() =>
                    copyPresetLink({
                      label: activeCase ? `${activeCase.title} 案例链接` : "当前案例链接",
                      nextDefenseMode: defenseMode,
                      nextCaseId: activeCaseId,
                    })
                  }
                />
              </div>
            )}
          </aside>
        </section>

        {!defenseMode && (
          <section className="support-strip">
            {quickFacts.map((item) => (
              <article className="support-card" key={item.title}>
                <span className="section-label">Guide</span>
                <strong>{item.title}</strong>
                <p>{item.body}</p>
              </article>
            ))}
          </section>
        )}
      </main>
    </div>
  );
}

function MetricChip({ label, children }) {
  return (
    <article className="metric-chip">
      <span>{label}</span>
      <strong>{children}</strong>
    </article>
  );
}

function InfoBlock({ title, children }) {
  return (
    <article className="info-block">
      <span>{title}</span>
      <strong>{children}</strong>
    </article>
  );
}

function ShareCard({ title, description, onCopy }) {
  return (
    <article className="share-card">
      <span className="section-label">Share</span>
      <strong>{title}</strong>
      <p>{description}</p>
      <button className="ghost-button compact" onClick={onCopy}>
        复制
      </button>
    </article>
  );
}

function TransformWorkspace({ matrix, probePoint, onProbeChange }) {
  return (
    <div className="transform-stage">
      <GraphPanel
        title="原始平面"
        mode="source"
        matrix={matrix}
        probePoint={probePoint}
        onProbeChange={onProbeChange}
        size={360}
      />
      <GraphPanel
        title="变换后平面"
        mode="target"
        matrix={matrix}
        probePoint={probePoint}
        size={360}
      />
    </div>
  );
}

function GraphPanel({ title, mode, matrix, probePoint, onProbeChange, size = 360 }) {
  const svgRef = useRef(null);
  const extent = 4;
  const center = size / 2;
  const unit = size / (extent * 2);
  const activePoint =
    mode === "source" ? probePoint : multiplyMatrixVector(matrix, probePoint);

  const toSvg = ([x, y]) => [center + x * unit, center - y * unit];
  const fromSvg = (clientX, clientY) => {
    const bounds = svgRef.current.getBoundingClientRect();
    const x = (clientX - bounds.left - center) / unit;
    const y = (center - (clientY - bounds.top)) / unit;
    return [clamp(x, -extent, extent), clamp(y, -extent, extent)];
  };

  const sourceBasis = [
    [1, 0],
    [0, 1],
  ];
  const targetBasis = sourceBasis.map((vector) =>
    multiplyMatrixVector(matrix, vector),
  );
  const basis = mode === "source" ? sourceBasis : targetBasis;

  const sourceSquare = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];
  const square =
    mode === "source"
      ? sourceSquare
      : sourceSquare.map((point) => multiplyMatrixVector(matrix, point));

  const gridLines = [];
  for (let i = -extent; i <= extent; i += 1) {
    const vertical = [
      [i, -extent],
      [i, extent],
    ];
    const horizontal = [
      [-extent, i],
      [extent, i],
    ];
    [vertical, horizontal].forEach((line) => {
      const actualLine =
        mode === "source"
          ? line
          : line.map((point) => multiplyMatrixVector(matrix, point));
      gridLines.push(actualLine);
    });
  }

  const pointSvg = toSvg(activePoint);
  const squarePath =
    square
      .map((point, index) => {
        const [x, y] = toSvg(point);
        return `${index === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .join(" ") + " Z";

  const startDrag = (event) => {
    if (mode !== "source" || !onProbeChange) return;
    const update = (moveEvent) => {
      onProbeChange(fromSvg(moveEvent.clientX, moveEvent.clientY));
    };
    update(event);
    window.addEventListener("pointermove", update);
    window.addEventListener(
      "pointerup",
      () => window.removeEventListener("pointermove", update),
      { once: true },
    );
  };

  return (
    <article className="graph-card">
      <div className="graph-card-header">
        <span className="section-label">{mode === "source" ? "Source" : "Target"}</span>
        <strong>{title}</strong>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${size} ${size}`}
        onPointerDown={startDrag}
      >
        <rect x="0" y="0" width={size} height={size} rx="26" className="graph-bg" />
        {gridLines.map((line, index) => {
          const [x1, y1] = toSvg(line[0]);
          const [x2, y2] = toSvg(line[1]);
          return (
            <line
              key={index}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              className="grid-line"
            />
          );
        })}
        <line x1={0} y1={center} x2={size} y2={center} className="axis-line" />
        <line x1={center} y1={0} x2={center} y2={size} className="axis-line" />
        <path d={squarePath} className="square-shape" />

        {basis.map((vector, index) => {
          const [x, y] = toSvg(vector);
          return (
            <g key={index}>
              <line
                x1={center}
                y1={center}
                x2={x}
                y2={y}
                className={`basis-line basis-${index}`}
              />
              <circle cx={x} cy={y} r="5" className={`basis-dot basis-${index}`} />
            </g>
          );
        })}

        <circle cx={pointSvg[0]} cy={pointSvg[1]} r="8" className="probe-dot" />
      </svg>
      <div className="graph-footer">
        {mode === "source"
          ? `P = (${roundNumber(probePoint[0], 2)}, ${roundNumber(
              probePoint[1],
              2,
            )})`
          : `T(P) = (${roundNumber(activePoint[0], 2)}, ${roundNumber(
              activePoint[1],
              2,
            )})`}
      </div>
    </article>
  );
}

export default App;

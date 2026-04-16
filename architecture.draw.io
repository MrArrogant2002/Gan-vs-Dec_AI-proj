<mxfile host="app.diagrams.net" modified="2026-04-15T12:00:00.000Z" agent="Codex" version="24.7.17" type="device" compressed="false">
  <diagram id="deepfake-robustness-arch" name="Architecture">
    <mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1800" pageHeight="1400" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <mxCell id="2" value="Deepfake Robustness Pipeline&#xa;Source of truth: agent-run.ipynb" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1F4E78;fontColor=#FFFFFF;strokeColor=#14344E;fontStyle=1;fontSize=20;" vertex="1" parent="1">
          <mxGeometry x="520" y="30" width="540" height="70" as="geometry" />
        </mxCell>

        <mxCell id="3" value="Data Sources" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E1D5E7;strokeColor=#9673A6;fontStyle=1;fontSize=16;" vertex="1" parent="1">
          <mxGeometry x="40" y="140" width="300" height="360" as="geometry" />
        </mxCell>
        <mxCell id="4" value="PubMed Abstracts Subset&#xa;Hugging Face streaming&#xa;Real label = 0" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F8CECC;strokeColor=#B85450;" vertex="1" parent="1">
          <mxGeometry x="75" y="200" width="230" height="80" as="geometry" />
        </mxCell>
        <mxCell id="5" value="Uploaded fake folders&#xa;fakenews_article / sentence&#xa;Fake label = 1" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;" vertex="1" parent="1">
          <mxGeometry x="75" y="300" width="230" height="80" as="geometry" />
        </mxCell>
        <mxCell id="6" value="Med-MMHL fallback repo&#xa;used if uploads are absent" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF2CC;strokeColor=#D6B656;" vertex="1" parent="1">
          <mxGeometry x="75" y="400" width="230" height="70" as="geometry" />
        </mxCell>

        <mxCell id="7" value="Primary Colab Notebook Pipeline" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontStyle=1;fontSize=16;" vertex="1" parent="1">
          <mxGeometry x="390" y="140" width="960" height="360" as="geometry" />
        </mxCell>
        <mxCell id="8" value="agent-run.ipynb&#xa;Colab controller&#xa;resume + orchestration" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="770" y="165" width="200" height="70" as="geometry" />
        </mxCell>
        <mxCell id="9" value="Dataset acquisition&#xa;download / upload / fallback" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="430" y="280" width="150" height="70" as="geometry" />
        </mxCell>
        <mxCell id="10" value="Cleaning + normalization&#xa;dedupe + filtering + split" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="610" y="280" width="150" height="70" as="geometry" />
        </mxCell>
        <mxCell id="11" value="GPT-2 Small&#xa;generator fine-tuning" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="790" y="280" width="150" height="70" as="geometry" />
        </mxCell>
        <mxCell id="12" value="BioBERT&#xa;baseline detector" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="970" y="280" width="150" height="70" as="geometry" />
        </mxCell>
        <mxCell id="13" value="BioGPT + LoRA&#xa;agent setup" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="1150" y="280" width="150" height="70" as="geometry" />
        </mxCell>
        <mxCell id="14" value="Adversarial round loop&#xa;5 rounds" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="720" y="395" width="170" height="70" as="geometry" />
        </mxCell>
        <mxCell id="15" value="Evaluation + plots&#xa;AUC / F1 / evasion rate" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="930" y="395" width="170" height="70" as="geometry" />
        </mxCell>
        <mxCell id="16" value="HF Hub export / reload&#xa;publish + sanity check" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="1140" y="395" width="170" height="70" as="geometry" />
        </mxCell>

        <mxCell id="17" value="Persistent Storage and Distribution" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#666666;fontStyle=1;fontSize=16;" vertex="1" parent="1">
          <mxGeometry x="1390" y="140" width="340" height="360" as="geometry" />
        </mxCell>
        <mxCell id="18" value="Google Drive&#xa;checkpoints / CSV / JSON / plots&#xa;metrics_log.csv&#xa;round_artifacts/*" style="shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;fillColor=#FFF2CC;strokeColor=#D6B656;" vertex="1" parent="1">
          <mxGeometry x="1450" y="220" width="220" height="110" as="geometry" />
        </mxCell>
        <mxCell id="19" value="Hugging Face Hub&#xa;gan-vs-det-ai-gpt2-generator&#xa;gan-vs-det-ai-biobert-detector&#xa;gan-vs-det-ai-biogpt-agent" style="shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;fillColor=#D5E8D4;strokeColor=#82B366;" vertex="1" parent="1">
          <mxGeometry x="1450" y="360" width="220" height="110" as="geometry" />
        </mxCell>

        <mxCell id="20" value="Adversarial Round Loop Detail" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E6F0FF;strokeColor=#6C8EBF;fontStyle=1;fontSize=16;" vertex="1" parent="1">
          <mxGeometry x="60" y="560" width="1670" height="430" as="geometry" />
        </mxCell>
        <mxCell id="21" value="1. Generate fake pool&#xa;GPT-2&#xa;pool size = 200" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="100" y="650" width="170" height="90" as="geometry" />
        </mxCell>
        <mxCell id="22" value="2. Score pool&#xa;with current detector" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="320" y="650" width="170" height="90" as="geometry" />
        </mxCell>
        <mxCell id="23" value="3. Select hard samples&#xa;Top-K = 50" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="540" y="650" width="170" height="90" as="geometry" />
        </mxCell>
        <mxCell id="24" value="4. Rewrite with&#xa;BioGPT + LoRA" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="760" y="650" width="170" height="90" as="geometry" />
        </mxCell>
        <mxCell id="25" value="5. Re-score rewrites&#xa;successful evasion if&#xa;score &lt; 0.5" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#6C8EBF;" vertex="1" parent="1">
          <mxGeometry x="980" y="650" width="180" height="90" as="geometry" />
        </mxCell>
        <mxCell id="26" value="6. Retrain detector&#xa;on augmented data" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;" vertex="1" parent="1">
          <mxGeometry x="1210" y="620" width="180" height="90" as="geometry" />
        </mxCell>
        <mxCell id="27" value="7. Fine-tune agent&#xa;on successful rewrites" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;" vertex="1" parent="1">
          <mxGeometry x="1210" y="760" width="180" height="90" as="geometry" />
        </mxCell>
        <mxCell id="28" value="8. Evaluate on test split&#xa;save metrics / rewrites / predictions" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="1460" y="685" width="220" height="100" as="geometry" />
        </mxCell>
        <mxCell id="29" value="Runtime constraints&#xa;- Google Colab Free&#xa;- staged model load / unload&#xa;- mixed precision&#xa;- LoRA to reduce BioGPT cost" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F8CECC;strokeColor=#B85450;" vertex="1" parent="1">
          <mxGeometry x="100" y="820" width="250" height="120" as="geometry" />
        </mxCell>
        <mxCell id="30" value="Reported notebook outputs&#xa;- AUC / F1 vs round&#xa;- Evasion rate vs round&#xa;- metrics_log.csv&#xa;- round_artifacts/*&#xa;- HF model repos" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E1D5E7;strokeColor=#9673A6;" vertex="1" parent="1">
          <mxGeometry x="430" y="820" width="280" height="120" as="geometry" />
        </mxCell>

        <mxCell id="31" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#666666;" edge="1" parent="1" source="4" target="9">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="32" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#666666;" edge="1" parent="1" source="5" target="9">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="33" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#666666;" edge="1" parent="1" source="6" target="9">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="34" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="8" target="9">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="35" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="9" target="10">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="36" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="10" target="11">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="37" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="10" target="12">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="38" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="12" target="13">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="39" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#82B366;" edge="1" parent="1" source="11" target="14">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="40" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#82B366;" edge="1" parent="1" source="12" target="14">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="41" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#82B366;" edge="1" parent="1" source="13" target="14">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="42" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="14" target="15">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="43" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="15" target="16">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="44" value="checkpoints / metrics" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#666666;" edge="1" parent="1" source="14" target="18">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="45" value="plots" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#666666;" edge="1" parent="1" source="15" target="18">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="46" value="upload" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#666666;" edge="1" parent="1" source="16" target="19">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="47" value="reload / test" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;startArrow=block;startFill=1;endArrow=block;endFill=1;strokeColor=#666666;" edge="1" parent="1" source="19" target="16">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <mxCell id="48" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="14" target="21">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="49" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="21" target="22">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="50" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="22" target="23">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="51" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="23" target="24">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="52" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#6C8EBF;" edge="1" parent="1" source="24" target="25">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="53" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#D79B00;" edge="1" parent="1" source="25" target="26">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="54" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#D79B00;" edge="1" parent="1" source="25" target="27">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="55" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#82B366;" edge="1" parent="1" source="26" target="28">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="56" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#82B366;" edge="1" parent="1" source="27" target="28">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="57" value="next round" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;dashed=1;strokeColor=#9673A6;" edge="1" parent="1" source="28" target="21">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="58" value="results" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#9673A6;" edge="1" parent="1" source="28" target="30">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="59" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#666666;" edge="1" parent="1" source="28" target="18">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>

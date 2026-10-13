"""Publication figures generated only from committed experiment artifacts."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .artifacts import WORK, read_json, digest, commit_json

BOX = dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5)
COLORS = {"original":"#B64342", "strong":"#0F4D92", "single_fault_lookup":"#42949E"}
LABELS = {"original":"Original", "strong":"Reference", "single_fault_lookup":"Lookup"}


def style():
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":9, "axes.linewidth":1.1, "axes.spines.top":False,
        "axes.spines.right":False, "pdf.fonttype":42, "ps.fonttype":42,
        "svg.fonttype":"none", "savefig.facecolor":"white"})


def protected_text(figure):
    # Protect every visible text object, including labels/ticks/legends.
    figure.canvas.draw()
    from matplotlib.text import Text
    for item in figure.findobj(match=Text):
        if item.get_visible() and item.get_text(): item.set_bbox(BOX)


def export(fig, name, inputs, caption):
    protected_text(fig)
    folder=WORK/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for suffix in ("pdf","png","svg"):
        path=folder/(name+"."+suffix)
        fig.savefig(path,dpi=350,bbox_inches="tight",pad_inches=.04)
        outputs.append(dict(path=str(path),sha256=digest(path)))
    plt.close(fig)
    commit_json(folder/(name+"-provenance.json"),dict(
        inputs={str(path):digest(path) for path in inputs},outputs=outputs,
        source_sha256=digest(__file__),caption=caption,
        text_boxes="all visible text protected by white background, alpha .85"))


def order_space():
    source=WORK/"results/order-space/result.json"
    comparisons=WORK/"results/order-space/comparisons.json"
    result=read_json(source)
    rows=[row for row in result["risks"] if not row["order"].startswith("anchor-")]
    assert len(rows)==64
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.52),gridspec_kw={"width_ratios":[1,1.05]})
    ax=axes[0]
    for name in ("original","strong"):
        ax.scatter([row["R_star"] for row in rows],[row["R_delta"][name] for row in rows],
            s=14,alpha=.55,color=COLORS[name],marker="o" if name=="original" else "^",label=LABELS[name])
    for name,offset in (("original",(6,8)),("strong",(6,8))):
        chosen=next(row for row in rows if row["order"]==result["winners"][name])
        ax.scatter(chosen["R_star"],chosen["R_delta"][name],marker="*",s=130,
            color=COLORS[name],edgecolor="black",linewidth=.6,zorder=5)
        ax.annotate(chosen["order"],(chosen["R_star"],chosen["R_delta"][name]),
            xytext=offset,textcoords="offset points",fontsize=7,bbox=BOX)
    ax.set(xlabel=r"Optimal single-fault risk $R^*$",ylabel=r"Implemented risk $R_\delta$")
    ax.legend(loc="upper right",framealpha=.85,facecolor="white",edgecolor="none",fontsize=8)
    ax.text(.02,1.03,"(a) 64 resource-matched orders",transform=ax.transAxes,bbox=BOX)
    ax=axes[1]
    selected=["anchor-forward","macro7-checks0","macro6-checks1"]
    names=["Forward","Original / optimal\nselection","Reference\nselection"]
    samples=[next(s for s in result["samples"] if s["task"]["order"]==name and s["task"]["eps"]==1e-4)
             for name in selected]
    x=np.arange(3)
    for j,name in enumerate(("original","strong","single_fault_lookup")):
        records=[sample["decoder"][name] for sample in samples]
        y=np.array([r["rate"]*100 for r in records]); ci=np.array([r["interval"] for r in records])*100
        ax.errorbar(x+(j-1)*.19,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=("o","^","s")[j],
            color=COLORS[name],capsize=2,ms=4,ls="none",label=LABELS[name])
    ax.set(xticks=x,xticklabels=names,ylabel="Logical failure (%)",ylim=(0,6.2))
    ax.tick_params(axis="x",labelsize=7)
    ax.text(.02,1.03,r"(b) Independent samples, $\epsilon=10^{-4}$",transform=ax.transAxes,bbox=BOX)
    ax.legend(loc="upper right",framealpha=.85,facecolor="white",edgecolor="none",fontsize=7)
    fig.tight_layout(w_pad=1.5)
    export(fig,"order-selection",[source,comparisons],
        "Single-fault risks over 64 orders and independently sampled failure for coefficient-selected orders. "
        "Stars identify the original- and reference-configuration minima; original and optimal-reference selection agree. "
        "Initial noisy extraction is fixed. Each confirmation uses 65,536 shots; bars show marginal 95% exact binomial intervals.")


def local_control():
    orders=("forward","macro_reverse","check_reverse")
    paths=[WORK/"results/joint-bridge"/order/"eps0.0001/result.json" for order in orders]
    native=WORK/"results/native/d3/all4-noisy-v3/eps0.0001/result.json"
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.5))
    rows=[read_json(path) for path in paths]
    x=np.arange(3)
    for j,(name,label) in enumerate((("full_joint_original","Original"),("full_joint_strong","Reference"),("single_fault_lookup","Lookup"))):
        color=list(COLORS.values())[j]
        axes[0].bar(x+(j-1)*.23,[row["decoder"][name]["R_delta"] for row in rows],.22,
            color=color,label=label,edgecolor="white",linewidth=.5)
    axes[0].set(xticks=x,xticklabels=["Forward","Macro reverse","Check reverse"],ylabel="Single-fault risk")
    axes[0].tick_params(axis="x",labelsize=7)
    axes[0].legend(loc="upper left",fontsize=7,framealpha=.85,edgecolor="none")
    axes[0].text(.02,1.03,"(a) Complete-DEM BP–OSD",transform=axes[0].transAxes,bbox=BOX)
    control=[read_json(native),*rows]; x=np.arange(4)
    for j,(name,label) in enumerate((("full_joint_original","Original"),("full_joint_strong","Reference"),("single_fault_lookup","Lookup"))):
        values=[row["decoder"][name] for row in control]
        y=np.array([row["failures"]/row["shots"]*100 for row in values]);ci=np.array([row["interval"] for row in values])*100
        axes[1].errorbar(x+(j-1)*.17,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=("o","^","s")[j],
            color=list(COLORS.values())[j],ls="none",ms=4,capsize=2,label=label)
    axes[1].set(xticks=x,xticklabels=["Native d3","Forward","Macro\nreverse","Check\nreverse"],ylabel="Logical failure (%)",ylim=(-.1,2.2))
    axes[1].tick_params(axis="x",labelsize=7)
    axes[1].text(.02,1.03,r"(b) Same decoder, $\epsilon=10^{-4}$",transform=axes[1].transAxes,bbox=BOX)
    fig.tight_layout(w_pad=2)
    export(fig,"complete-dem-control",[*paths,native],
        "Complete-detector-model decoding of the primary bridge orders and the native distance-three surface-code control. "
        "All four extraction blocks carry SD6 noise, and final data readout is ideal. "
        "Each circuit uses 32,768 shots; bars show marginal 95% exact binomial intervals.")


if __name__ == "__main__":
    style(); order_space(); local_control()
    print("two figures exported with input hashes",flush=True)

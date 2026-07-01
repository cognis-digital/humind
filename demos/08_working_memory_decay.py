"""Scenario 8 - attention dynamics (cognitive-science students).

Working memory is small, and it forgets. Each item carries an **activation** that
decays every tick; re-encountering an item re-activates it; and once activation
falls below a threshold the item drops out of focus entirely. This demo drives a
tiny capacity-3 buffer tick by tick with an activation bar, so you can watch
attention shift, an item get rehearsed back to the top, and stale items fade.
"""
from _common import bar, rule
from humind import WorkingMemory


def show(wm, label):
    print(f"\n{label}")
    if not wm.items:
        print("   (empty)")
    for it in sorted(wm.items, key=lambda i: i.activation, reverse=True):
        print(f"   {it.content:<10} {bar(it.activation)} {it.activation:.2f}")


def main() -> None:
    rule("WORKING MEMORY  -  a small, decaying attention buffer (ACT-R / Miller)")
    wm = WorkingMemory(capacity=3, decay=0.7, threshold=0.25)
    print(f"\ncapacity={wm.capacity}  decay={wm.decay}  threshold={wm.threshold}")

    wm.add("vessel", 1.6); wm.add("radar", 0.8); wm.add("corridor", 1.0)
    show(wm, "1) three contacts enter focus:")

    wm.add("drone", 1.4)                         # a fourth arrives -> evicts the weakest
    show(wm, "2) a 4th contact arrives; capacity forces an eviction:")

    wm.tick()
    show(wm, "3) one tick of decay (x0.7):")

    wm.add("vessel", 1.2)                        # rehearse 'vessel' back up
    show(wm, "4) 'vessel' is mentioned again -> re-activated to the top:")

    wm.tick(); wm.tick()
    show(wm, "5) two more ticks; items below threshold fall out of focus:")

    print("\nNothing is deleted by hand: capacity, decay and threshold together")
    print("produce a naturally shifting focus — the current 'now' of the mind.")


if __name__ == "__main__":
    main()

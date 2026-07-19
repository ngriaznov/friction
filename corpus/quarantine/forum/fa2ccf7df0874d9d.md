Test Tools
=====

In approximate descending order of safety (though the professional multimeter could arguably be higher if you are careful about how you use it)

Voltage tester
--------------

A professional electrician would very likely have one of these:

[![Fluke T90 voltage tester][1]][1]

If you expect to do occasional DIY electrical work in the rest of your life, you should consider buying something like this.

- It does not make you part of the circuit.
- It has no knobs or switches that can accidentally be left in the wrong position.
- It has no removable leads that can be accidentally left in the wrong socket.
- it can *easily* be used with two hands (more so than a typical multimeter)
- It will operate without batteries.
- It is relatively inexpensive (&lt;&#163;25 in UK).
- It is a major brand - so the design can be expected to incorporate high standards for safety and reliability.
- It will likely last the rest of your life.

If money is an issue you can get cheaper versions of the above as shop-brand items (e.g. cheapest CPC/Farnell &quot;Tenma&quot; branded device &lt;&#163;8), this is what I use now.

[![CPC Voltage tester][2]][2]

----

2-in-1 tester
---------------------

At the cheapest extreme you might consider one of these instead:

[![eBay 2-in-1 voltage tester][4]][4]

- It does not make you part of the test circuit
- It will work without batteries
- Cheap (~ &#163;4 from eBay vendors)

It does have some disadvantages

- It is very cheaply made.
- The LED indicators are a bit dim and hard to see, especially at lower voltages.

I&#39;m not sure I trust it fully. It is certainly safer, and a better buy, than a &quot;voltage test screwdriver&quot; (see below).

---

Non-Contact Voltage (NCV) detector
--------------------------

Sometimes called a &quot;voltage detector pen&quot; or just &quot;voltage detector&quot;

[![Extech DVA30 voltage and current detector][5]][5]

This type of detector is widely used for the following reasons

- Inexpensive (the one pictured is more expensive as it also detects current).
- No contact with wiring needed.
  - So does not make you part of the circuit.
- few or no controls to set (so fewer mistakes possible)

It has some disadvantages

- Can be misleading and &quot;detect&quot; voltage several inches (or more) from the wires
- Not very good for detecting which specific wire is live in a cable
- Battery required
- Can be fiddly to verify correct operation.

The one I have has a thumb-wheel that has to be turned to set the sensitivity, you need to do this each time you use it and a consequence is that the test result isn&#39;t necessarily as clear and distinct as the other types of tester.

Some electricians say that NCV detectors are good for an initial test but that, for safety, you should follow up with a contact based voltage detector like the one at the top of this answer.

See https://diy.stackexchange.com/q/14080/2815

---

Multimeter (professional)
-------------------------

I have several of these bought cheaply from eBay (around &#163;70).

[![Fluke 77-IV multimeter][6]][6]

Advantages:

- It is credibly Cat-II rated .
- The leads are also Cat-II rated.
- It has a four-socket design, so current measurements are physically separated from voltage measurements.
- It contains a good HRC fuse (two actually).
- It has lots of other protections (MOVs etc).
- The case is sufficiently well-designed and made to contain the small electrical explosions that might occur when you make a bad mistake.

**Hold**

A nice feature with this meter is that the &quot;Hold&quot; button is a true hold and not a mere &quot;data hold&quot;. This is useful for those of use with fewer than three arms.

With a true hold, you can put the multimeter on the floor (or hang it nearby), press the hold button, then take a probe in each hand and make the measurement without looking at the meter display, you stop taking the measurement when the meter beeps, you take your hands away from the wiring then look at the meter to see what value it shows.

With a &quot;data hold&quot;, you have to use a third hand and your other set of eyes to press the hold button during the reading. In practice it can mean using one hand to hold both probes, looking away from what your hands are doing near the wiring (dangerous) and pressing the hold button.

So the former (rarer) type of &quot;hold&quot; is useful, the latter not.

**Low-Z**

A problem using multimeters to test for voltage is that they typically have high impedance, typically 10 MΩ and can pick up &quot;ghost voltages&quot;. These are induced voltages caused by proximity of wires. They are not dangerous voltages as no significant current can be supplied *inductively* in typical household wiring. Some meters have a Low-Z setting designed to prevent this measurement artifact.

**Disadvantages**

Like all multimeters it has some significant disadvantages

- You can accidentally try to measure volts with the leads plugged into the current sockets (some meters have a jack-alert feature to warn about this)
- You can accidentally try to measure volts with the knob set to the wrong range (perhaps making you think wires are safe when they are not)

---

Multimeter (cheap)
------------

I have several of these type of cheap multimeters
[![Caltek CM1200 multimeter][7]][7]

Even though this one is apparently rated Cat-II I prefer not to use it to test for 120V/240V. Because

- The case is not designed to contain electric arcs.
- The fuse is a cheap fuse not a high rupture capacity fuse.
- It has a 3-socket design. The same socket is used for volts and current.

All these things mean more likelihood of unpleasant injury when you make a mistake.

---

Voltage Test Screwdriver
-------------------------

Most folk have one of these

[![Voltage test screwdrivers][8]][8]

The bottom type contains only a small 1/4W resistor and a neon indicator.

The top type is battery powered and can be used for a variety of purposes: mains voltage, continuity test, finding the break in a cable, etc.

[![Voltage test screwdriver showing continuity function][9]][9]

Both of them are fundamentally unsafe because

- They make you part of the 120V/240V test circuit.

This is bad. Even though they are cheap and millions of people use them. Buy the 2-in-1 tester if money is an issue.

---

Test Tools Summary
-------

Use a reputable simple voltage-tester, a really decent multimeter or a 2-in-1 type tester. An NCV detector is useful for initial tests but be careful not to rely on it too much. Don&#39;t use a tester that makes you part of the circuit and don&#39;t use a cheap multimeter.

---

Other Tools
===========

Most DIYers own ordinary screwdrivers, pliers, wire-cutters and strippers. It can be safer to buy insulated tools intended to protect you when working on 120V/240V AC electrical wiring.

There are different standards bodies which makers test their products against. In Europe it is normal to look for VDE rated tools. In the US I guess the equivalent is UL. This should be stamped on the tool (but be wary of cheap unbranded or off-brand tools made in countries that have a poor record for faking such marks)

Insulated screwdriver
----------------------

[![Generic insulated screwdriver][10]][10]

Often a low-cost set of these will include a voltage test screwdriver (see above) which I suggest be discarded.

Insulated wire-cutter/stripper
------------------------------

[![VDE-rated Wire cutters by NWS][11]][11]

Note that this one combines wire-cutting and insulation-stripping features

Insulated Pliers
------

[![CK insulated pliers][12]][12]

Note that the insulation is rated for 1000 volts and that the handles have protruding guards to help stop your fingers slipping onto the metal conductive parts.

---

Electrical test tool categories
===

Test equipment is rated and marked as Cat-II etc -

So Cat-II is OK for most electrical work around the home - changing light-fittings, light-switches and sockets/outlets. For working inside the main electrical panel, or near to it, Cat-III would be more appropriate.

---

Techniques
==========

See https://diy.stackexchange.com/questions/47613/electric-shock-was-i-stupid-unlucky-or-a-combination-of-both

The main principles are

- turn off the circuit using the circuit-breaker at the main electrical panel. It is prudent to mark, tape or even lock the circuit breaker switch so that someone else doesn&#39;t switch it back on while you are working on the wiring.
- keep your hands away from bare metal (wires etc). This is why good probes have prominent finger-guards and probe-tip shields.
- check and recheck that your test equipment
   - leads are plugged into appropriate sockets
   - knobs are set to appropriate ranges
   - other features of the test equipment are set appropriately
- After a test confirms a wire is dead, check the equipment has not just now failed. Test it on a known live source before proceeding to handle the wiring.

The third point above is why multimeters are not an ideal tester for 120V/240V work.

  [1]: https://i.sstatic.net/YcZt4.png
  [2]: https://i.sstatic.net/argmh.jpg
  [3]: https://i.sstatic.net/uvJi8.png
  [4]: https://i.sstatic.net/JX3L0.png
  [5]: https://i.sstatic.net/HU1j8.jpg
  [6]: https://i.sstatic.net/P172Q.jpg
  [7]: https://i.sstatic.net/rk1WP.jpg
  [8]: https://i.sstatic.net/CeUo4.jpg
  [9]: https://i.sstatic.net/2j48I.jpg
  [10]: https://i.sstatic.net/ZZFR4.jpg
  [11]: https://i.sstatic.net/6WzMl.png
  [12]: https://i.sstatic.net/tv8Ct.jpg
  [13]: https://i.sstatic.net/hsnLj.png

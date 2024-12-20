from typing import Tuple
from manim import *
import numpy as np
from nn import NueralNetwork
import math


class moving(Scene):
    colorGrad = ["#ffffff",
                    "#dfdfdf",
                    "#c0c0c0",
                    "#a2a2a2",
                    "#858585",
                    "#696969",
                    "#4e4e4e",
                    "#353535",
                    "#1e1e1e",
                    "#000000"]
    
    def getMaxes(self, weights,perNode):
        maxs = self.setList(None,[perNode for _ in weights])
        for i in range(len(self.nn)-1):
            for j in range(self.nn[i]):
                maxv = [float('-inf')] * perNode
                for idx in range(self.nn[i+1]):
                    w = weights[i][idx][j]
                    for p in range(perNode):
                        if self.nn[i+1]/self.nn[i] < 3 and w > maxv[p]:
                            for k in range(perNode - 1, p, -1):
                                maxv[k] = maxv[k - 1]
                            maxv[p] = w
                            break
                maxs[i][j] = maxv
        return maxs
    def setWeights(self,wieghtsPerNode,animate:bool):
        maxs = self.getMaxes(self.nueralNet.weights,3)
        for i in range(len(self.nn)-1):
            for j in range(self.nn[i]):
                for k in range(self.nn[i+1]):
                    val = self.nueralNet.weights[i][k][j]
                    line = Line(self.nuerons[i][j].get_end(),self.nuerons[i+1][k].get_start() - (0.4,0,0), color = self.getColor(val))
                    self.weights[i][k][j] = (line,val)
                    if val < maxs[i][j][wieghtsPerNode-1]:
                        continue
                    self.netShapes.add(line)
                    if animate:
                        self.play(Create(line,run_time=0.3))
                    else:
                        self.add(line)
    
    def removeWeights(self,obj):
        if isinstance(obj, list):
            for i in obj:
                self.removeWeights(i)
        else:
            # print(obj[0])
            self.remove((obj[0]))

    def getColor(self, num):
        
        return ManimColor(self.colorGrad[max(min(9,math.ceil(num*10)),0)])

    def transformWeights(self,newWeights):
        maxs = self.getMaxes(newWeights,3)
        for i in range(len(self.nn)-1):
            layer = []
            for j in range(self.nn[i]):
                for k in range(self.nn[i+1]):
                    weight = self.weights[i][k][j][0]
                    val = newWeights[i][k][j]
                    self.weights[i][k][j] = (weight,val)
                    if val < maxs[i][j][-1]:
                        continue
                    layer.append(weight.animate.set_color(color = self.getColor(val)))
            self.play(layer,run_time=0.2)



    def setList(self,val, dims):
        # If dims is empty, just return the value
        if len(list(dims)) == 0:
            return val
        
        # If dims is a single integer, return a list of that many 'val'
        if isinstance(dims, int):
            return [val for _ in range(dims)]
        
        # If dims is a list, we have two cases:
        # 1. A list of integers (homogeneous dimensions)
        # 2. A list that may contain integers and/or further lists (non-homogeneous)
        
        if all(isinstance(d, int) for d in dims):
            if len(list(dims)) == 1:
                return [val for _ in range(dims[0])]
            else:
                return [self.setList(val, dims[1:]) for _ in range(dims[0])]
        else:

            result = []
            for d in dims:
                result.append(self.setList(val, d))
            return result
        


    config.background_color = DARK_GRAY
    def construct(self):
        config.background_color = LOGO_WHITE
        self.nn = [1,3,3,1]
        self.netShapes = VGroup()
        self.colorGrad.reverse()
        self.nueralNet = NueralNetwork(self.nn,"tanh",activateLast=False)
        wieghtsPerNode = 3
        self.weights:List[List[List[Tuple[Mobject,float]]]] = [[[None]*i] for i in self.nn]
        self.nuerons:List[List[Mobject]] = []
        self.weights = self.setList(None,[x.shape for x in self.nueralNet.weights])

        for i in range(len(self.nn)):
            self.nuerons.append([])
            for _ in range(self.nn[i]):
                self.nuerons[i].append(Circle(color = WHITE,fill_color= GRAY_BROWN,fill_opacity=0.7, radius = 0.2,).move_to((0,3,0)))

        

        print(self.weights)
        for layer in self.nuerons:
            for nueron in layer:
                self.add(nueron)
                self.netShapes.add(nueron)


        # for i in range(len(self.nn)):
        #     layer = []
        #     for j in range(self.nn[i]):
        #         x = (i - (len(self.nn) - 1) * 0.5) * (0.1 + 2)
        #         y = (j - (self.nn[i] - 1) * 0.5) * (0.2 + 0.3)
        #         layer.append(self.nuerons[i][j].animate.move_to([x,y,0]))
        #     self.play(*layer)

        for i in range(len(self.nn)):
            layer = []
            for j in range(self.nn[i]):
                x = (i - (len(self.nn) - 1) * 0.5) * (0.1 + 2)
                y = (j - (self.nn[i] - 1) * 0.5) * (0.2 + 0.3)
                layer.append(self.nuerons[i][j].move_to([x,y,0]))
            self.add(*layer)


        # for i in range(len(nn)-1):
        #     waves = []
        #     for j in range(nn[i]):
        #         for k in range(nn[i+1]):
        #             if nn[i+1]/nn[i] < 3 and weights[i][j][k][1] < maxs[i][j][wieghtsPerNode-1]:
        #                 continue
        #             dir = weights[i][j][k][0].get_start() - weights[i][j][k][0].get_end()
        #             waves.append(ApplyWave(weights[i][j][k][0],direction=(-dir[1],dir[0],0),run_time=1))
        #     self.play(waves)

        # self.wait()
        # self.removeWeights(self.weights)
        decay = 0

        interWeights = []
        def train():
            examples = 200
            inputs = []
            for i in range(examples):
                inputs.append(np.random.rand(self.nn[0],1)*4)
            epochs = 1000
            outputs = [np.floor(input) for input in inputs]
            loss = [0]
            for epoch in range(epochs+1):
                learningRate = 0.06*(1/(1+decay*epoch))
                
                def prop(x):
                    return self.nueralNet.propegateFull(x)
                prop = np.vectorize(prop)
                if (epoch % (epochs//10)) == 0:
                    print("Epoch: ", epoch, "Loss: ", loss)
                if epoch == 0:
                    interWeights.append((self.nueralNet.weights.copy(),[0],0,axes.plot(lambda x:  self.nueralNet.propegateFull(np.array([[x]]))[0][0], color=BLUE)))
                else:
                    interWeights.append((self.nueralNet.weights.copy(),loss,epoch,axes.plot(lambda x:  self.nueralNet.propegateFull(np.array([[x]]))[0][0], color=BLUE)))
            
                loss = self.nueralNet.batch(inputs,outputs,1,learningRate)


        
        # self.transformWeights(self.nueralNet.weights)
        self.wait()
        anim = []
        if False:
            self.setWeights(3,animate=True)
            self.play(self.netShapes.animate.scale(0.8))
            self.play(self.netShapes.animate.shift(LEFT*3.5))
        else:
            self.setWeights(3,animate=False)
            self.add(self.netShapes.scale(0.8))
            self.add(self.netShapes.shift(LEFT*3.5))
        # [self.play(ani) for ani in anim] 

        axes = Axes(
            x_range=[0, 4, 0.5],  # x-axis range with tick step
            y_range=[0, 5, 1],  # y-axis range with tick step
            axis_config={"include_numbers": True},  # Show numbers on the axes
        )
        axes_labels = axes.get_axis_labels(x_label="", y_label="")
        # graph_label = axes.get_graph_label(graph, label="Output")
        
        plot_group = VGroup(axes, axes_labels)
        self.wait(0.5)
        self.play(plot_group.animate.scale(0.5))
        self.play(plot_group.animate.shift(RIGHT*2.6))
        update_label = Tex("Initial random Weights").next_to(self.netShapes,UP*3)

        identityGraph = (DashedVMobject(axes.plot(lambda x: np.floor(x),color=RED)))
        colorSquare = Square(0.2,stroke_width=0,fill_color=RED,fill_opacity=1).next_to(axes,UP*4).shift(LEFT*2.2).shift(DOWN*1.5)
        colorLabel = Tex("$y = \\text{floor}(x)$").scale(0.3).next_to(colorSquare,RIGHT*0.5)
        self.play(FadeIn(colorSquare,colorLabel),Create(identityGraph))
        self.wait(1)
        self.play(FadeIn(update_label))
        self.wait(0.5)

        train()
        epochsToPlay =  [i*20 for i in range(0,5)] + [i for i in range(100,len(interWeights),100)] + [len(interWeights)-1]
        prevLoss = 0
        
        label = Tex("")
        graph = None
        colorSquare = Square(0.2,stroke_width=0,fill_color=BLUE,fill_opacity=1).next_to(colorSquare,DOWN)
        colorLabel = Tex("$y = \\text{Nureal\\_Net}(x)$").scale(0.3).next_to(colorSquare,RIGHT*0.5)
        self.play(FadeIn(colorSquare,colorLabel))
        
        for i in epochsToPlay:
            weights, loss,epoch, currGraph = interWeights[i]
            learningRate = 0.2*(1/(1+decay*i))

            if i != 0:
                update_label = Tex(r'$Update  Weights\;\uparrow$').next_to(self.netShapes,UP*3)
                self.play(FadeIn(update_label))
                self.transformWeights(weights)
                self.wait(0.1)
                self.remove(graph)
            self.play(FadeOut(update_label),run_time=0.2)
            self.play(Transform(label,Tex(f"Epoch:\#{epoch},  Loss: {loss[0]*10**3:0.2f}m  Change: {(loss[0] - prevLoss)*10**3:.2f}m,  Learning Rate: {learningRate:.2f}").next_to(plot_group,UP*3,buff=0.2).scale(0.4)),run_time=0.2)

            self.play(Create(currGraph),run_time=0.1)
            prevLoss = loss[0]
            graph = currGraph
            # self.remove(label)

        # d = nuerons[0][0]
        # d.move_to([-3, 2, 0])

        # self.play(Write(d))
        # self.play(d.animate.move_to([5, -1, 0]))
        # self.play(d.animate.move_to(ORIGIN))
        # self.play(Unwrite(d))
var ws = null;
var handler = null;

//{group, x, y, count, distance, rel_size, repr_text}
summary_table = null;
//{id, label, children, count, parent}
summary_tree = null;
//{text, x, y, predict, choice1, choice2, choice3, distance, distance2, distance3}
predict_result = null;
vis_data = null;
//[id]
group_filter = null;

viewing_trainset = true;

function UploadHandler(file, fname, onComplete, onError) {
    this.onCom = onComplete;
    this.onErr = onError;
    this.f = file;
    this.reader = new FileReader();
    this.uploaded=0;
    this.nextChunk=0;
    this.chunkSize=512*1024;
    
    this.reader.onload = (e) => {
        param = {
            "data": e.target.result,
            "flag": this.uploaded===0?"begin":"continue",
            "filename": fname,
        }
        this.uploaded += this.nextChunk;
        $(".prgs:last").html("Uploading " + this.uploaded + "/" + this.f.size).detach().appendTo("#logs");
        send_action("upload", param);
    }
    
    this.start = () => {
        if (ws.readyState != WebSocket.OPEN) {
            log("err", "Not connected");
            return;
        }
        log("prgs", "Uploading");
        this.next(0);
    }
    
    this.next = (r) => {
        blob = this.f.slice(this.uploaded, this.uploaded + this.chunkSize);
        this.nextChunk = blob.size;
        if (blob.size) {
            this.reader.readAsDataURL(blob);
        } else {
            param = {
                "size": this.f.size,
                "flag": "end",
                "filename": fname,
            }
            send_action("upload", param);
        }
    }
    
    this.success = () => {
        $(".prgs:last").html("Uploaded " + this.uploaded + "/" + this.f.size);
        this.onCom(this);
    }
    
    this.error = (r) => {
        this.onErr(this, r["result"]);
    }
}

function TableUpdateHandler() {
    this.success = () => {
        handler = null;
        table = $("#dataset_table").DataTable();
        $("#tb_dataset").click();
        table.ajax.reload();
    }
}

str_to_int_array = (str) => {
    var ints = [];
    str = str.split(",");
    for (idx in str) {
        s = str[idx];
        try {
            let g = s.match(/([0-9]+)\-([0-9]+)/);
            if (g) {
                let start = Number(g[1]);
                let end = Number(g[2]);
                for (var i = start; i <= end; i++)
                    ints.push(i);
            } else {
                ints.push(Number(s));
            }
        } catch (e) {
        }
    }
    return ints;
}
filter_group = (row, filter) => {
    if (typeof(filter)!=="object") return true;
    if (typeof(filter["idx"])==="string")
        filter["idx"] = str_to_int_array(filter["idx"]);
    if ((typeof(filter["idx"])==="object")&&(filter["idx"].length)) {
        if (filter["idx"].indexOf(row.idx) == -1) return false;
    }
    if ((typeof(filter["text"])==="string")&&(filter["text"] !== "")) {
        if (row.text.toLowerCase().indexOf(filter["text"].toLowerCase()) == -1)
            return false;
    }
    return true;
}

//#region visualize scatter
create_scatter_table = (filterStr) => {
    //try to load data from server and restart
    if (!vis_data || !(vis_data["train"] && summary_table)) {
        let toolbar = $("#visualizepane .toolbar").html("");
        let text_filter = $("<input type='text' id='visfilter_text' placeholder='Filter (Enter to apply)'>");
        text_filter.val(filterStr).appendTo(toolbar);
        $("#visualize_scatters").click();
        return;
    }
    var x = [];
    var y = [];
    //var z = [];
    var s = [];
    var o = [];
    var m = [];
    var g = [];
    let c1 = [0, 1, 1];
    let c2 = [300, 1, 1];
    let filter;
    let showgroups = $("#visfilter_show_groups").length ? $("#visfilter_show_groups").prop("checked") : true;
    let showTrains = $("#visfilter_show_trains").length ? $("#visfilter_show_trains").prop("checked") : true;
    let showTests = $("#visfilter_show_tests").length ? $("#visfilter_show_tests").prop("checked") : true;
    let showAdhocs = $("#visfilter_show_adhocs").length ? $("#visfilter_show_adhocs").prop("checked") : true;
    let hideFiltered = $("#visfilter_hide_filtered").length ? $("#visfilter_hide_filtered").prop("checked") : true;

    let t;
    if (filterStr) {
        t = filterStr;
    } else {
        t = $("#visfilter_text").val();
    }
    if (t && t.startsWith("predict:")) {
        let idx = t.replace("predict:", "");
        filter = {"idx": idx};
    } else {
        filter = {"text": t};
    }
    
    $("#tb_visualize").click();
    
    if (showgroups && summary_table != null) {
        summary_table.forEach((v, _) => {
            not_filtered = filter_group({"idx": v.group, "text": v.repr_text}, filter);
            if (!not_filtered && hideFiltered) return;
            x.push(v.x);
            y.push(v.y);
            //z.push(v.z);
            s.push(Math.max(20, v.distance * 100));
            if (showTrains || showTests || showAdhocs)
                o.push(0.3);
            else
                o.push(0.8);
            m.push(['group', v.group, 'Sample', v.repr_text, 'Size', v.distance]);
            color = mix_color_hsv(c1, c2, v.group / summary_table.length);
            if (not_filtered)
                g.push(color);
            else 
                g.push('#ddd');
        });
    }
    if (showTrains && (vis_data != null) && (vis_data["train"] != null)) {
        vis_data["train"].forEach((v, _) => {
            not_filtered = filter_group({"idx": v.predict, "text": v.text}, filter);
            if (!not_filtered && hideFiltered) return;
            x.push(v.x);
            y.push(v.y);
            //z.push(v.z);
            s.push(3);
            if (showAdhocs)
                o.push(0.6);
            else
                o.push(1);
            m.push(['Train', v.text, 'Ref', v.ref, 'group', v.predict]);
            if (v.predict == -1) {
                g.push('#aaa');
            } else {
                color = mix_color_hsv(c1, c2, v.predict / summary_table.length);
                if (not_filtered)
                    g.push(color);
                else 
                    g.push('#ddd');
            }
        });
    }
    if (showTests && (vis_data != null) && (vis_data["dev"] != null)) {
        vis_data["dev"].forEach((v, _) => {
            not_filtered = filter_group({"idx": v.predict, "text": v.text}, filter);
            if (!not_filtered && hideFiltered) return;
            x.push(v.x);
            y.push(v.y);
            s.push(3);
            if (showAdhocs)
                o.push(0.6);
            else
                o.push(1);
            m.push(['Test', v.text, 'Ref', v.ref, 'group', v.predict]);
            if (v.predict == -1) {
                g.push('#aaa');
            } else {
                color = mix_color_hsv(c1, c2, v.predict / summary_table.length);
                if (not_filtered)
                    g.push(color);
                else 
                    g.push('#ddd');
            }
        });
    }
    if (showAdhocs && (predict_result != null)) {
        predict_result.forEach((v, _) => {
            not_filtered = filter_group({"idx": v.choice1, "text": v.text}, filter);
            if (!not_filtered && hideFiltered) return;
            x.push(v.x);
            y.push(v.y);
            //z.push(v.z);
            s.push(3);
            o.push(1);
            m.push(['Adhoc', v.text, 'group', v.choice1, 'Distance', v.distance]);
            color = mix_color_hsv(c1, c2, v.choice1 / summary_table.length);
            if (not_filtered)
                g.push(color);
            else 
                g.push('#ddd');
        });
    }
    var data = [{
        x: x,
        y: y,
        //z: z,
        marker: {
            opacity: o,
            size: s,
            color: g,
        },
        mode: 'markers',
        type: 'scatter',
        hovertemplate: '<b>%{meta[0]}: %{meta[1]}</b><br><br>' +
                       '<b>%{meta[2]}:</b> %{meta[3]}<br>' +
                       '<b>%{meta[4]}:</b> %{meta[5]}<br>',
        meta: m,
    }];
    var layout = {
        showlegend: false,
        title: "Texts and groups projections"
    }
    current_graph = $("#visualization_chart").data("current_graph");
    if (current_graph == "scatter") {
        Plotly.react("visualization_chart", data, layout);
    } else {
        $("#visualization_chart").data("current_graph", "scatter").html("");
        Plotly.newPlot("visualization_chart", data, layout);
        $("#visualization_chart")[0].on("plotly_click", (e) => {
            if (e.points[0].meta[0] == "group") {
                group = e.points[0].meta[1];
                check_group(group);
            } else if (e.points[0].meta[0] == "Train") {

            } else if (e.points[0].meta[0] == "Adhoc") {
                $("#show_predict_dialog").click();
            }
        }).on("plotly_relayout", (e) => {

        });
        
        let toolbar = $("#visualizepane .toolbar").html("");
        let text_filter = $("<input type='text' id='visfilter_text' placeholder='Filter (Enter to apply)'>");
        text_filter.keydown((e) => {
            if ((e.keyCode == 13)||(e.originalEvent.keyCode == 13))
                create_scatter_table();
        });
        text_filter.appendTo(toolbar);
        ['show_groups', 'show_trains', 'show_tests', 'show_adhocs', 'hide_filtered'].forEach((v, _) => {
            let chkbox = $("<input type='checkbox' name='visfilter_" + v + "' id='visfilter_" + v + "' checked>");
            chkbox.change((e) => {
                create_scatter_table();
            });
            chkbox.appendTo(toolbar);
            $("<label for-'visfilter_" + v + "'>" + v.replace("_", " ") + "</label>").appendTo(toolbar);
        })
    }
    if (t) $("#visfilter_text").val(t);
}
//#endregion

//#region visualize group over time
create_timeline = (filterStr) => {
    //load data from server side first
    if (!vis_data || !vis_data["counts"]) {
        let toolbar = $("#visualizepane .toolbar").html("");
        let text_filter = $("<input type='text' id='visfilter_text' placeholder='Filter (Enter to apply)'>");
        text_filter.val(filterStr).appendTo(toolbar);
        $("#visualize_timeline").click();
        return;
    }
    let t;
    if (filterStr) {
        t = filterStr;
    } else {
        t = $("#visfilter_text").val();
    }
    let filter;
    if (t && t.startsWith("predict:")) {
        let idx = t.replace("predict:", "");
        filter = {"idx": idx};
    } else {
        filter = {"text": t};
    }
    var data = [];
    vis_data["counts"].forEach((v, _) => {
        not_filtered = filter_group({"idx": v["idx"], "text": v["text"]}, filter);
        if (!not_filtered) return;
        data.push({
            x: v.x,
            y: v.y,
            name: "Group " + v.idx,
            text: v.text,
            meta: v.idx,
        });
    });
    var layout = {
        title: "Group trends",
    }

    $("#tb_visualize").click();

    current_graph = $("#visualization_chart").data("current_graph");
    if (current_graph == "timeline") {
        Plotly.react("visualization_chart", data, layout);
    } else {
        $("#visualization_chart").data("current_graph", "timeline").html("");
        Plotly.newPlot("visualization_chart", data, layout);
        $("#visualization_chart")[0].on("plotly_click", (e) => {
            if (e.points[0]) {
                group = e.points[0].data.meta;
                check_group(group);
            }
        });
        
        let toolbar = $("#visualizepane .toolbar").html("");
        let text_filter = $("<input type='text' id='visfilter_text' placeholder='Filter (Enter to apply)'>");
        text_filter.keydown((e) => {
            if ((e.keyCode == 13)||(e.originalEvent.keyCode == 13))
                create_timeline();
        });
        text_filter.appendTo(toolbar);
    }
    if (t) $("#visfilter_text").val(t);

}
//#endregion

//#region visualize group_count
visualize_group_count = (filter) => {
    create_group_count_table("visualization_chart", filter);
    let text_filter = $("<input type='text' id='visfilter_group_text' placeholder='filter'>");
    text_filter.keydown(() => {
        let t = $("#visfilter_group_text").val();
        if (t.startsWith("predict:")) {
            let idx = t.replace("predict:", "");
            let filter = {"idx": idx};
            create_group_count_table("visualization_chart", filter);
        } else {
            let filter = {"text": t};
            create_group_count_table("visualization_chart", filter);
        }
    });
    $("#visualizepane .toolbar").html("");
    text_filter.appendTo($("#visualizepane .toolbar"));
    $("#tb_visualize").click();
}

create_group_count_table = (tgt, filter) => {
    let count_table = summary_table.sort((a, b) => {return a.count < b.count});
    var x = [];
    var y = [];
    var m = [];
    var g = [];
    count_table.forEach((v, i) => {
        x.push(i);
        y.push(v.count);
        m.push([v.group, v.repr_text]);
        if (filter_group({"idx": v.group, "text": v.repr_text}, filter))
            g.push('#88f');
        else 
            g.push('#ddd');
    });
    var data = [{
        x: x,
        y: y,
        marker: {color: g},
        type: 'bar',
        hovertemplate: '<b>group %{meta[0]}</b><br>' +
                       '%{meta[1]}<br><br>' +
                       '<b>Count</b> %{y}<br>' +
                       '<b>Rank</b> %{x}<br>',
        meta: m,
    }];
    var layout = {
        showlegend: false,
        bargap: 0,
        title: "Groups ordered by count"
    }
    current_graph = $("#" + tgt).data("current_graph");
    if (current_graph == "group_count") {
        Plotly.react(tgt, data, layout);
    } else {
        $("#" + tgt).data("current_graph", "group_count").html("");
        Plotly.newPlot(tgt, data, layout);
        $("#" + tgt)[0].on("plotly_click", (e) => {
            group = e.points[0].meta[0];
            check_group(group);
        });
    }
}
//#endregion

//#region groups manipulation
group_table_filter = (settings, data, dataIdx) => {
    if (settings.nTable.id != "groups_table")
        return true;
    if (!group_filter)
        return true;
    return (group_filter.indexOf(Number(data[1])) != -1);
}

expand_collpase_tree = async (node, $tree) => {
    adjust_node = (node, $tree, result) => {
        let jqtnode = $tree.tree("getNodeById", node.id);
        if (result) {
            $tree.tree("openNode", jqtnode, false);
            $(jqtnode.element).css({"opacity": 1});
        } else {
            $tree.tree("closeNode", jqtnode, false);
            $(jqtnode.element).css({"opacity": 0.4});
        }
    }
    return new Promise((resolve)=> {
        work = () => {
            var result = false;
            if (node.children) {
                promises = [];
                for (var i = 0; i < node.children.length; i++) {
                    v = node.children[i];
                    promises.push(expand_collpase_tree(v, $tree));
                }
                Promise.all(promises).then((r) => {
                    result = r.reduce((a,b)=>a | b);
                    adjust_node(node, $tree, result);
                    resolve(result);
                })
            } else {
                result = (group_filter.indexOf(node.id) != -1);
                adjust_node(node, $tree, result);
                resolve(result);
            }
        }
        setTimeout(work, 0);
    });
}

// // input node: node in summary_tree
// expand_collpase_tree = (node, $tree) => {
//     var result = false;
//     if (node.children) {
//         node.children.forEach((v)=> {
//             if (expand_collpase_tree(v, $tree)) result = true;
//         })
//     } else {
//         result = (group_filter.indexOf(node.id) != -1);
//     }
//     let jqtnode = $tree.tree("getNodeById", node.id);
//     if (result) {
//         $tree.tree("openNode", jqtnode, false);
//         $(jqtnode.element).css({"opacity": 1});
//     } else {
//         $tree.tree("closeNode", jqtnode, false);
//         $(jqtnode.element).css({"opacity": 0.4});
//     }
//     return result;
// }
filter_group_tree = async () => {
    if (!group_filter) return;
    let $tree = $("#groups_tree");
    loading_dialog("Filtering...");
    setTimeout(()=> {
        expand_collpase_tree(summary_tree, $tree);
        hide_loading_dialog();    
    }, 500);
}
function groupsUpdateHandler() {
    create_chart = (r) => {
        $("<div id='clusteredpie' class='charts'/>").appendTo($("#group_charts"))[0];
        $("<div id='groupcount' class='charts'/>").appendTo($("#group_charts"))[0];
        var data = [{
            labels: ["ungrouped", "grouped"],
            values: [r.stats.no_group, r.stats.has_group],
            type: 'pie',
        }];
        var layout = {
            height: 400,
            width: 400,
        }
        Plotly.newPlot('clusteredpie', data, layout);
        create_group_count_table('groupcount');
    }
    
    this.error = (r) => {
        handler = null;
        $("#groups_table_wrapper").hide();
        $("#groups_tree").hide();
        $("#modelpane .charts").hide();
        $("#modelpane .placeholder").show();
    }
    this.success = (r) => {
        handler = null;
        
        summary_table = r.groups;
        summary_tree = r.tree;

        $("#modelpane .placeholder").hide();

        $("#groups_table_wrapper").show();
        table = $("#groups_table").DataTable();
        table.rows().remove();

        summary_table.forEach((group) => {
            table.rows.add([[group["group"], group["group"], group["repr_text"], group["count"], group["distance"]]]);
        });
        table.on("draw", ()=>{
            map_check_group();
        });
        table.draw();
        $("#tb_model").click();
        $("#groups_table_wrapper").show();
        setTimeout(() => {
            $("#groups_table").DataTable().columns.adjust();
            map_check_group();
        }, 100);
        $("#modelpane .charts").remove();
        loading_dialog("Creating chart and graphs...")
        log("msg", "Creating chart and graphs...");
        create_chart(r);
        setTimeout(() => {
            $("#groups_tree").remove();
            $('<div id="groups_tree">').appendTo($("#group_list"));
            $tree = $("#groups_tree");
            $("#groups_tree").hide();
            $tree.tree({
                data: [r.tree],
                autoOpen: true,
                onCreateLi: (node, li, is_selected) => {
                    if (node.id < 0) {
                        return;
                    }
                    chk = $('<input type="checkbox" class="chk" data-id="' + node.id + '" value="' + node.id + '">');
                    if (is_selected) chk.prop('checked', true);
                    chk.click((e) => {
                        e.stopPropagation();
                    })
                    li.children("div").prepend(chk);
                },
            }).on("tree.click", (e) => {
                e.preventDefault();
                var selected_node = e.node;
                if (selected_node.id < 0) {
                    return;
                }
                check_group(selected_node.id);
            });
            log("msg", "Ready...");
            hide_loading_dialog();
        }, 200);
        
    }
}
//#endregion

show_prediction_result = () => {
    //todo: present result
    //text,x,y,z,predict,choice1,choice2,choice3,distance,distance2,distance3
    if (!predict_result || !summary_table) return;
    $table = $("#predict_result").DataTable();
    $table.rows().remove();

    predict_result.forEach((result) => {
        predict = result["choice1"];
        confidence = 1 - result["distance"] / summary_table[predict]["distance"];
        if (confidence > 1) confidence = 1;
        //else if (confidence < 0) confidence = 0;  
        $table.rows.add([[result["text"], predict, result["distance"], confidence]]);
    });
    $table.draw();
    map_check_group();
    setTimeout(()=>{
        $("#show_predict_dialog").click();
    }, 500);
}
update_result = () => {
    if (handler) return;
    handler = new groupsUpdateHandler()
    send_action("read_train_result");
}

download = function(path) {
    if ($("#ws_dummy").length == 0)
        $('<iframe src="about:blank" id="ws_dummy"></iframe>').appendTo("#logs");
    $("#ws_dummy").attr("src", path);
}
log = function(cls, msg) {
    if (!msg) return;
    oldmsg = $("#logs div:last").html()
    if (oldmsg == msg) return;
    $("<div class='" + cls + "'>" + msg + "</div>")
        .appendTo("#logs");
    $('#logs').scrollTop($('#logs').prop("scrollHeight"));
}

send_action = function(action, param) {
    if (!ws) return;
    if (!param) {
        param = {};
    }
    param["action"] = action;
    ws.send(JSON.stringify(param));
}

//#region Upload related functions

csv_preview = (file, table) => {
    table.empty(); // clean table
    reader = new FileReader();
    reader.onload = () => {
        let alllines = $.csv.toArrays(reader.result);
        let line0 = alllines[0];
        let abcexport = line0[0] == "#ABCExportv1";
        let linefrom;
        if (abcexport)
            linefrom = 2;
        else
            linefrom = 0;
        let lines;
        lines = alllines.slice(linefrom, linefrom+5);
        let numitems = lines[0].length;
        let thead = $('<thead/>').appendTo(table);
        let tr = $("<tr/>").appendTo(thead);
        for (i = 0; i < numitems; i++) {
            $th = $("<th/>").html(`
            <select class="dataset_rowtype" style="width: 100%;" data-idx="` + i + `">
                <option value='ignore'>Ignore</option>
                <option value='text'>Text</option>
                <option value='ref'>Reference</option>
                <option value='ts'>Timestamp</option>
                <option value='target'>Target</option>
                <option value='predict'>Predicted</option>
                <option value='distance'>Distance</option>
                <option value='set'>Set</option>
            </select>`);
            if (abcexport) {
                $th.find("option:nth(" + (i+1) + ")").prop("selected", true);
            } else {
                $th.find("option:first").prop("selected", true);
            }
            $th.appendTo(tr);
        }
        let tbody = $('<tbody/>').appendTo(table);
        for (i = 0; i < lines.length; i++) {
            let tr = $("<tr/>").appendTo(tbody);
            for (j = 0; j < numitems; j++) {
                try {
                    $("<td/>").html(lines[i][j].trim()).appendTo(tr);
                } catch(e) {
                    $("<td/>").appendTo(tr);
                }
            }
        }
        if (abcexport) {
            $("#has_comment").prop("checked", true);
            $("#ignore_line").prop("checked", true);
            $("#ignore_line_count").val(2);
        }
    }
    reader.readAsBinaryString(file.slice(0, 16384)); //16K preview
}

do_upload_csv = (f) => {
    _param = {
        "type": "csv",
        "filename": "uploaded.csv",
    };
    $(".dataset_rowtype").each((_i, v) => {
        if ($(v).val()=="text") _param["text"] = $(v).data("idx");
        else if ($(v).val()=="target") _param["target"] = $(v).data("idx");
        else if ($(v).val()=="predict") _param["predict"] = $(v).data("idx");
        else if ($(v).val()=="ts") _param["ts"] = $(v).data("idx");
        else if ($(v).val()=="ref") _param["ref"] = $(v).data("idx");
        else if ($(v).val()=="distance") _param["distance"] = $(v).data("idx");
        else if ($(v).val()=="set") _param["set"] = $(v).data("idx");
    })
    if ($("#has_comment").prop("checked")) {
        _param["comment"] = "#";
    }
    if ($("#ignore_line").prop("checked")) {
        _param["skipfirst"] = $("#ignore_line_count").val();
    }
    if ((_param["text"] === null)||(_param["text"] < 0)) {
        log("err", 'Please select at least the text column');
        return;
    }
    handler = new UploadHandler(f, "uploaded.csv", () => {
        handler = new TableUpdateHandler();
        send_action("load_dataset", _param);
    }, () => {
        handler = null;
    });
    handler.start();
}
json_preview = (file, table) => {
    table.empty(); // clean table
    reader = new FileReader();
    reader.onload = () => {
        rows = JSON.parse(reader.result);
        numrows = rows.length;
        if (numrows > 5) numrows = 5;
        keys = Object.keys(rows[0]);
        thead = $('<thead/>').appendTo(table);
        tr = $("<tr/>").appendTo(thead);
        for (i = 0; i < keys.length; i++) {
            $("<th/>").html(`
            <select class="dataset_rowtype" style="width: 100%;" data-idx="` + keys[i] + `">
                <option value='ignore' selected>Ignore</option>
                <option value='text'>Text</option>
                <option value='ref'>Reference</option>
                <option value='ts'>Timestamp</option>
                <option value='target'>Target</option>
                <option value='predict'>Predicted</option>
                <option value='distance'>Distance</option>
            </select>`).appendTo(tr);
        }
        tbody = $('<tbody/>').appendTo(table);
        for (i = 0; i < numrows; i++) {
            tr = $("<tr/>").appendTo(tbody);
            for (j = 0; j < keys.length; j++) {
                try {
                    $("<td/>").html(rows[i][keys[j]]).appendTo(tr);
                } catch(e) {
                    $("<td/>").appendTo(tr);
                }
            }
        }
    }
    reader.readAsText(file);
}
do_upload_json = (f) => {
    _param = {
        "type": "json",
        "filename": "uploaded.json",
    };
    $(".dataset_rowtype").each((_i, v) => {
        if ($(v).val()=="text") _param["text"] = $(v).data("idx");
        else if ($(v).val()=="target") _param["target"] = $(v).data("idx");
        else if ($(v).val()=="predict") _param["predict"] = $(v).data("idx");
        else if ($(v).val()=="ts") _param["ts"] = $(v).data("idx");
        else if ($(v).val()=="ref") _param["ref"] = $(v).data("idx");
        else if ($(v).val()=="distance") _param["distance"] = $(v).data("idx");
    })
    if ((_param["text"] === null)||(_param["text"] < 0)) {
        log("err", 'Please select at least the text column');
        return;
    }
    handler = new UploadHandler(f, "uploaded.json", () => {
        handler = new TableUpdateHandler();
        send_action("load_dataset", _param);
    }, () => {
        handler = null;
    });
    handler.start();
}
do_upload_sqlite = (f) => {
    handler = new UploadHandler(f, "uploaded.sqlite", () => {
        handler = new TableUpdateHandler();
        param = {
            "filename": "uploaded.sqlite",
            "type": "sqlite",
            "table": $("#sql_table").val(),
            "text": $("#sql_text").val(),
        }

        for (key in ["embedding", "umap", "x", "y", "z", "ref", "ts", "target", "predict"]) {
            if ($("#sql_has_" + key).val()) {
                param[key] = $("#sql_" + key).val();
            }
        }
        send_action("load_dataset", param);
    }, () => {
        handler = null;
    });
    handler.start();
}
do_upload_model = (f) => {
    handler = new UploadHandler(f, "uploaded.model", () => {
        handler = new TableUpdateHandler();
        param = {
            "filename": "uploaded.model",
        }
        send_action("load_model", param);
    }, () => {
        handler = null;
    });
    handler.start();
}
//#endregion

//#region group dialog
check_group = (group_id) => {
    handler = {
        success: (r) => {
            handler = null;
            $("#tinfo_id").html(r.group.group);
            $("#tinfo_count").html(r.group.count);
            $("#tinfo_coverage").html(r.group.distance);
            $("#tinfo_repr_text").html(r.group.repr_text);
            $("#tinfo_neighbours").html("");
            for (id in r.neighbours) {
                neighbour = r.neighbours[id];
                div = $("<div class='tinfo_neighbour'>").appendTo($("#tinfo_neighbours"));
                $("<p><span class='infoline_name'>ID</span><span class='infoline_value tinfo_neighbour_id'>" + neighbour.group + "</span></p>").appendTo(div);
                $("<p><span class='infoline_name'>Distance</span><span class='infoline_value'>" + neighbour.distance + "</span></p>").appendTo(div);
                $("<p><span class='infoline_name'>Similaritiy</span><span class='infoline_value'>" + neighbour.score + "</span></p>").appendTo(div);
                $("<p><span class='infoline_name'>Sample</span><span class='infoline_value'>" + neighbour.repr_text + "</span></p>").appendTo(div);
            }
            $("#filter_ds").off("click").click(() => {
                $("#tb_dataset").click();
                $.modal.close();
                setTimeout(() => {
                    $("#dataset_table").DataTable().search("predict:"+$("#tinfo_id").html()).draw();
                }, 100);
            });                        
            $("#filter_ds_with_neighbour").off("click").click(() => {
                groups = [$("#tinfo_id").html()];
                $(".tinfo_neighbour_id").each((i, v) => {
                    groups.push($(v).html());
                });
                $.modal.close();
                $("#tb_dataset").click();
                setTimeout(() => {
                    $("#dataset_table").DataTable().search("predict:"+groups.join(",")).draw();
                }, 100);
            });
            $("#locate_ds").off("click").click(() => {
                $.modal.close();
                create_scatter_table("predict:" + $("#tinfo_id").html());
            });
            $("#locate_ds_with_neighbour").off("click").click(() => {
                $.modal.close();
                groups = [$("#tinfo_id").html()];
                $(".tinfo_neighbour_id").each((i, v) => {
                    groups.push($(v).html());
                });
                create_scatter_table("predict:" + groups.join(","));
            });
            $("#ds_trends").off("click").click(() => {
                $.modal.close();
                create_timeline("predict:" + $("#tinfo_id").html());
            });
            $("#ds_trends_with_neighbour").off("click").click(() => {
                $.modal.close();
                groups = [$("#tinfo_id").html()];
                $(".tinfo_neighbour_id").each((i, v) => {
                    groups.push($(v).html());
                });
                create_timeline("predict:" + groups.join(","));
            });
            $("#hb_groupinfo").click();
        }
    };
    params = {
        "group": group_id,
    }
    send_action("get_group_info", params);
}
//#endregion

//#region Filter and search
create_filter_string = () => {
    var filterstr = [];
    ["text", "ref", "ts_before", "ts_exact", "ts_after", "predict", "target"].forEach((v) => {
        if ($("#filter_by_" + v).prop("checked")) {
            val = $("#filter_" + v).val();
            if (!val) return;
            if (v == "text") {
                filterstr.push(val);
            } else if (v == "ts_before") {
                filterstr.push("ts:<" + val);
            } else if (v == "ts_exact") {
                filterstr.push("ts:" + val);
            } else if (v == "ts_after") {
                filterstr.push("ts:>" + val);
            } else 
                filterstr.push(v + ":" + val);
        }
    });
    return filterstr.join("&&");
}
do_group_search = () => {
    if (handler) return;
    handler = {
        success: (r) => {
            handler = null;
            group_filter = r["groups"];
            $("#groups_table").DataTable().draw(); //will search
            filter_group_tree();
            create_group_count_table('groupcount', {"idx": group_filter});
        }
    }
    text = $("#group_search").val();
    send_action("group_search", {
        "filter": text
    });
}
//#endregion

function getHexColor(number){
    return "#"+((number)>>>0).toString(16).slice(-6);
}
hsv2rgb = function (hsb) {
    var rgb = { };
    var h = hsb[0];
    var s = hsb[1];
    var v = hsb[2];

    if (s == 0) {
        rgb.r = rgb.g = rgb.b = v;
    } else {
        var t1 = v;
        var t2 = (1 - s) * v;
        var t3 = (t1 - t2) * (h % 60) / 60;

        if (h == 360) h = 0;

        if (h < 60) { rgb.r = t1; rgb.b = t2; rgb.g = t2 + t3 }
        else if (h < 120) { rgb.g = t1; rgb.b = t2; rgb.r = t1 - t3 }
        else if (h < 180) { rgb.g = t1; rgb.r = t2; rgb.b = t2 + t3 }
        else if (h < 240) { rgb.b = t1; rgb.r = t2; rgb.g = t1 - t3 }
        else if (h < 300) { rgb.b = t1; rgb.g = t2; rgb.r = t2 + t3 }
        else if (h < 360) { rgb.r = t1; rgb.g = t2; rgb.b = t1 - t3 }
        else { rgb.r = 0; rgb.g = 0; rgb.b = 0 }
    }

    return { r: Math.round(rgb.r*255), g: Math.round(rgb.g*255), b: Math.round(rgb.b*255) }
}
function rgb2hsv (rgb) {
    var r = rgb[0];
    var g = rgb[1];
    var b = rgb[2];

    var computedH = 0;
    var computedS = 0;
    var computedV = 0;
   
    r=r/255; g=g/255; b=b/255;
    var minRGB = Math.min(r,Math.min(g,b));
    var maxRGB = Math.max(r,Math.max(g,b));
   
    // Black-gray-white
    if (minRGB==maxRGB) {
     computedV = minRGB;
     return [0,0,computedV];
    }
   
    // Colors other than black-gray-white:
    var d = (r==minRGB) ? g-b : ((b==minRGB) ? r-g : b-r);
    var h = (r==minRGB) ? 3 : ((b==minRGB) ? 1 : 5);
    computedH = 60*(h - d/(maxRGB - minRGB));
    computedS = (maxRGB - minRGB)/maxRGB;
    computedV = maxRGB;
    return [computedH,computedS,computedV];
}
mix_color_hsv = (hsv1, hsv2, a1) => {
    a2 = 1 - a1;
    let hsv = [hsv1[0] * a1 + hsv2[0] * a2, 
                hsv1[1] * a1 + hsv2[1] * a2,
                hsv1[2] * a1 + hsv2[2] * a2];
    let rgb = hsv2rgb(hsv);
    rs = ("00" + rgb.r.toString(16)).slice(-2);
    gs = ("00" + rgb.g.toString(16)).slice(-2);
    bs = ("00" + rgb.b.toString(16)).slice(-2);
    return "#" + rs + gs + bs;
}
mix_color = (c1, c2, a1) => {
    let rgb1 = [(c1 >> 16) & 255, (c1 >> 8) & 255, c1 & 255];
    let rgb2 = [(c2 >> 16) & 255, (c2 >> 8) & 255, c2 & 255];
    let hsv1 = rgb2hsv(rgb1);
    let hsv2 = rgb2hsv(rgb2);
    return mix_color_hsv(hsv1, hsv2, a1);
}

dstable_show_text_similarity = (s) => {
    if (!s) {
        $("#dataset_table tr").each((i, v) => {
            $(v).css({
                "background-color": ''
            });
        });
    } else {
        $("#dataset_table tr").each((i, v) => {
            id = $(v).find(".chk").data("id");
            val = s[id];
            if (!val) color = '';
            else {
                if (val < 0) val = 0;
                else if (val > 1) val = 1;
                c1 = 0xCCFFCC;
                c2 = 0xFFCCCC;
                color = mix_color(c1, c2, val);
            }
            $(v).css({
                "background-color": color
            });
        });
    }
}
function TextSimilarityHandler() {
    this.error = (r) => {
        handler = null;
        dstable_show_text_similarity();
    }
    this.success = (r) => {
        handler = null;
        dstable_show_text_similarity(r["similarities"]);
    }
}
map_check_group = () => {
    $(".check_group").off("click").click((e) => {
        group_id=$(e.delegateTarget).data("group");
        check_group(group_id);
    });
    $("#dataset_table tr").off("mouseover").on("mouseover", (e) => {
        if (handler) return;

        y = $(e.delegateTarget).find(".chk").data("id");
        if (!y) return;
        x = [];
        $("#dataset_table tr").each((i, v) => {
            id = $(v).find(".chk").data("id");
            if ((id!==undefined) && (id != y)) x.push(id);
        });
        handler = new TextSimilarityHandler();
        param = {
            "x": x,
            "y": y,
            "set": viewing_trainset ? "train" : "dev",
        };
        send_action("text_similarity", param);
    }).off("mouseout").on("mouseout", (e) => {
        dstable_show_text_similarity();
    });
}

dropzones = () => {
    document.querySelectorAll(".dropzone_input").forEach((inputElement) => {
        const dropZoneElement = inputElement.closest(".dropzone");

        dropZoneElement.addEventListener("click", (e) => {
            inputElement.click();
        });

        dropZoneElement.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZoneElement.classList.add("dropzone_over");
        });

        ["dragleave", "dragend"].forEach((type) => {
            dropZoneElement.addEventListener(type, (e) => {
                dropZoneElement.classList.remove("dropzone_over");
            });
        });

        dropZoneElement.addEventListener("drop", (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) {
                inputElement.files = e.dataTransfer.files;
                $(inputElement).change();
            }
            dropZoneElement.classList.remove("dropzone_over");
        });
    });
}

function trimchar(str, ch) {
    var start = 0, 
        end = str.length;

    while(start < end && str[start] === ch)
        ++start;

    while(end > start && str[end - 1] === ch)
        --end;

    return (start > 0 || end < str.length) ? str.substring(start, end) : str;
}

get_dataset_checked = () => {
    idx=[];
    $("#dataset_table .chk:checked").each((i, v) => {
        idx.push($(v).data("id"));
    });
    return idx;
}

close_menu = () => {
    $("#menu").removeClass("visiblemenu");
}

loading_dialog = (msg, progress=false) => {
    $("#loading_msg").html(msg.replace("Remote: ", ""));
    $("#loading_bar").progressbar("value", progress);
    let dialog = $("#loading_dialog");
    if (!dialog.is(":visible")) {
        dialog.modal({
            closeExisting: true,
            escapeClose: false,
            clickClose: false,
            showClose: false,
        });
    }
}
hide_loading_dialog = () => {
    $.modal.close();
    // setTimeout(()=>{
    //     $(".jquery-modal.blocker").remove();
    // }, 1000);
}
init = function() {

    $("#loading_bar").progressbar({
        max: 1,
    })
//#region WebSocket creation

    log("msg", "Websocket connect...");
    ws = new WebSocket('ws://' + location.host + '/ws');
    
    ws.onopen = function() {
        log("msg", 'Connected.');
        setTimeout(() => {send_action("ping")}, 100);
    };
    
    ws.onmessage = function(evt) {
        msg = JSON.parse(evt.data);
        if (msg["info"] && msg["info"]["name"]) {
            $("#model_name").html(msg["info"]["name"]);
        }
        if (msg["result"] < 0) {
            hide_loading_dialog();
            if (msg["message"]) log("err", "Remote: " + msg["message"]);
            if (handler && handler.error) {
                handler.error(msg);
            } else {
                handler = null;
            }
        } else if (msg["result"] == 0) { //completed
            hide_loading_dialog();
            if (msg["message"]) log("stream0", "Remote: " + msg["message"]);
            if (handler && handler.success) {
                handler.success(msg);
            } else {
                handler = null;
            }
        } else if ((msg["result"] == 4)||(msg["result"] == 1)) { 
            //1:info, 4:Logger
            loading_dialog(msg["message"]);
            log("stream1", "Remote: " + msg["message"]);
            if (handler && handler.next) {
                handler.next(msg["result"]);
            }
        } else if ((msg["result"] == 2)||(msg["result"] == 3)) { 
            //2: stdout, 3: stderr
            let progress = msg["progress"];
            var pval = false;
            if (progress) 
                pval = progress;

            stream = "stream" + msg["result"];
            msg = trimchar(msg["message"],"'").replace("\\t", "    ").split("\\n");
            first = msg[0].trim();
            if (first) {
                if ($("." + stream + ":last").length > 0) {
                    ele = $("." + stream + ":last");
                    oldmsg = ele.html();
                    if (first.includes("\\r")) {
                        v = first.split("\\r");
                        oldmsg = "Remote: " + v[v.length-1]
                    } else {
                        oldmsg = oldmsg + " " + first;
                    }
                    ele.html(oldmsg).detach().appendTo("#logs");
                    if (oldmsg != "") loading_dialog(oldmsg, pval);
                } else {
                    log(stream, "Remote: " + first);
                    if (first != "") loading_dialog(first, pval);
                }
            }
            for (i = 1; i < msg.length; i++) {
                m = msg[i].trim().split("\\r");
                m = m[m.length - 1];
                log(stream, "Remote: " + m);
                if (m != "") loading_dialog(m);
            }
        } 
    };
    ws.onerror = function(e) {
        log("err", "Error: " + e.message);
    }
    ws.onclose = function() {
        log("err", 'Connection closed.');
        loading_dialog("Connection closed, please manually refresh to reconnect.")
    };
//#endregion

    dropzones();
    $("#tabs").tabs();

//#region Menu related

    $(".menubtn").click(()=>{
        $menu = $("#menu");
        if ($menu.hasClass("visiblemenu")) {
            $menu.removeClass("visiblemenu");
        } else {
            $menu.addClass("visiblemenu");
        }
    });
    $("#menu .toolbar a").click(() => {
        close_menu();
    });
    $("#model_name").click(() => {
        if (handler) return;
        let name = prompt("Rename current model:", $("#model_name").html());
        if (name) {
            handler = null;
            send_action("rename_model", {"name": name});
        }
    });
    $.modal.defaults.closeExisting = false;
    $.modal.defaults.fadeDuration = 200;
    $.modal.defaults.fadeDelay = 0;
//#endregion

//#region Datatables
    //group table search function
    $.fn.dataTable.ext.search.push(group_table_filter);

    $("#dataset_table").DataTable({
        "serverSide": true,
        "ajax": {
            "url":"/ds/train",
            "type":"POST",
        },
        columns: [
            {data: 'id'},
            {data: 'id'},
            {data: 'text'},
            {data: 'ref'},
            {data: 'ts'},
            {data: 'target'},
            {data: 'predict'},
        ],
        columnDefs: [
            {
                targets: 0,
                render: function (data, type, full, meta){
                    return '<input type="checkbox" class="chk" data-id="' + full['id'] + '" value="' + full['id'] + '">';
                }
            },
            {
                targets: 6,
                render: function (data, type, full, meta){
                    if (full['predict'] !== '')
                        return '<a class="check_group" href="#" data-group="' + full['predict'] + '">' + full['predict'] + '</a>';
                    return '';
                }
            },
            {
                targets:[0,1,2,3,4,5,6],
                searchable: false,
                orderable: false,
            },
        ],
        search: {
            return: true
        },
        "lengthMenu": [ [10, 25, 50, 100, -1], [10, 25, 50, 100, "All"] ]
    }).on("draw", () => {
        $("#dataset_table").DataTable().columns.adjust();
        map_check_group();
        //map_show_embeddings();
    });
    $("#dataset_chkall").change((e) => {
        $("#dataset_table input[type=checkbox]").prop("checked", $("#dataset_chkall").prop("checked"));
    })
    $("#groups_table").DataTable({
        //searching: false,
        columnDefs: [
            {
                targets: 0,
                render: function(data, type, full, meta) {
                    return '<input type="checkbox" class="chk" data-id="' + full[0] + '" value="' + full[0] + '">';
                }
            },                        
            {
                targets: 1,
                render: function(data, type, full, meta) {
                    return '<a class="check_group" href="#" data-group="' + full[1] + '">' + full[1] + '</a>';
                }
            },
            {
                targets: 4,
                render: function(data, type, full, meta) {
                    return parseFloat(data).toFixed(4);
                }
            },
        ],
    })
    $("#predict_result").DataTable({
        searching: false,
        columnDefs: [
            {
                targets: 1,
                render: function(data, type, full, meta) {
                    return '<a class="check_group" href="#" data-group="' + data + '">' + data + '</a>';
                }
            },
            {
                targets: 2,
                render: function(data, type, full, meta) {
                    return parseFloat(data).toFixed(4);
                }
            },
        ],
    });
    
//#endregion

//#region Filter dialog
    $("#remove_filter").click(()=>{
        ["text", "ref", "ts_before", "ts_exact", "ts_after", "predict", "target"].forEach((v)=>{
            $("#filter_by_" + v).prop("checked", false);
        });
    });
    $("#filter_dataset").click(() => {
        $("#confirm_filter").off("click").click(()=> {
            $.modal.close();
            $("#dataset_table_filter input").val(create_filter_string());
            $("#dataset_table").DataTable().draw();
        });
        $("#filter_dialog").modal();
    });
    $("#filter_group").click(() => {
        $("#confirm_filter").off("click").click(()=> {
            $.modal.close();
            $("#group_search").val(create_filter_string());
            do_group_search();
        });
        $("#filter_dialog").modal();
    });
//#endregion

//#region Upload dialog

     $("#file_dataset").change(() => {
        try {
            let file = $("#file_dataset")[0].files[0];
            let filepath = file.name;
            let filename = filepath.split('\\').pop().split('/').pop();
            let fileext = filename.split(".").pop().toLowerCase();
            $("#dataset_file_prompt").html(filename);
            switch (fileext) {
                case 'model':
                    $("#upload_csv").hide();
                    $("#upload_sqlite").hide();
                    break;
                case 'csv':
                    $("#upload_csv").show();
                    $("#upload_sqlite").hide();
                    csv_preview(file, $("#dataset_preview"));
                    break;
                case 'json':
                    $("#upload_csv").show();
                    $("#upload_sqlite").hide();
                    json_preview(file, $("#dataset_preview"))
                    break;
                case 'sqlite':
                case 'sqlite3':
                    $("#upload_csv").hide();
                    $("#upload_sqlite").show();
                    break;
            }
           
        } catch (e) {
            log("err", e);
        }
    });
    $("#upload_sqlite input[type=checkbox]").change((e) => {
        let $target = $(e.delegateTarget);
        let id = $target.attr("id");
        let targetid = id.replace("sql_has_", "#sql_");
        $(targetid).prop("disabled", !$target.val())
    });
    $("#confirm_upload_dataset").click(()=>{
        $.modal.close();
        let file = $("#file_dataset")[0].files[0];
        let fileext = file.name.split(".").pop().toLowerCase();
        switch (fileext) {
            case 'model':
                    return do_upload_model(file);
                case 'csv':
                    return do_upload_csv(file);
                case 'json':
                    return do_upload_json(file);
                case 'sqlite':
                case 'sqlite3':
                    return do_upload_sqlite(file);
        }
    });
//#endregion

//#region Create model dialog
    $('#new_model_dialog input[type="text"]').attr('placeholder', "(empty for default)");
    $('#new_model_new_embed').change(() => {
        if ($("#new_model_new_embed").prop("checked")) {
            $("#new_model_embed").css({
                "max-height": "1000px",
                "opacity": "unset"
            });
        } else {
            $("#new_model_embed").css({
                "max-height": "0",
                "opacity": "0"
            });
        }
    });
    $('#new_model_new_umap').change(() => {
        if ($("#new_model_new_umap").prop("checked")) {
            $("#new_model_umap").css({
                "max-height": "1000px",
                "opacity": "unset"
            });
        } else {
            $("#new_model_umap").css({
                "max-height": "0",
                "opacity": "0"
            });
        }
    });
    $('#new_model_new_hdbscan').change(() => {
        if ($("#new_model_new_hdbscan").prop("checked")) {
            $("#new_model_hdbscan").css({
                "max-height": "1000px",
                "opacity": "unset"
            });
        } else {
            $("#new_model_hdbscan").css({
                "max-height": "0",
                "opacity": "0"
            });
        }
    });

    $("#confirm_create_model").click(() => {
        $.modal.close();
        if (handler) return;
        params = {};
        if ($("#new_model_new_embed").prop("checked")) {
            params["embedding"] = {};
            $("#new_model_embed input, #new_model_embed select").each((i, v) => {
                key=$(v).attr("id").replace("embed_", "");
                val=$(v).val();
                if (val) params["embedding"][key] = val;
            });
        }
        if ($("#new_model_new_umap").prop("checked")) {
            params["umap"] = {};
            $("#new_model_umap input, #new_model_umap select").each((i, v) => {
                key=$(v).attr("id").replace("umap_", "");
                val=$(v).val();
                if (val) params["umap"][key] = val;
            });
        }
        if ($("#new_model_new_hdbscan").prop("checked")) {
            params["hdbscan"] = {};
            $("#new_model_hdbscan input, #new_model_hdbscan select").each((i, v) => {
                key=$(v).attr("id").replace("hdbscan_", "");
                val=$(v).val();
                if (val) params["hdbscan"][key] = val;
            });
        }
        handler = {}
        send_action("create_model", params);
    });
//#endregion

//#region Train dialog

    $("#train_reset_embeddings").change(()=>{
        if ($("#train_reset_embeddings").prop("checked")) {
            $("#train_reset_reduction").prop("checked", true);
            $("#train_reset_clustering").prop("checked", true);
            $("#train_reset_projections").prop("checked", true);
        }
    });
    $("#train_reset_reduction").change(()=>{
        if ($("#train_reset_reduction").prop("checked")) {
            $("#train_reset_clustering").prop("checked", true);
            $("#train_reset_projections").prop("checked", true);
        } else {
            $("#train_reset_embeddings").prop("checked", false);
        }
    });
    $("#train_reset_clustering").change(()=>{
        if ($("#train_reset_clustering").prop("checked")) {
            $("#train_reset_projections").prop("checked", true);
        } else {
            $("#train_reset_embeddings").prop("checked", false);
            $("#train_reset_reduction").prop("checked", false);
        }
    });
    $("#train_reset_projections").change(()=>{
        if (!$("#train_reset_projections").prop("checked")) {
            $("#train_reset_embeddings").prop("checked", false);
            $("#train_reset_reduction").prop("checked", false);
            $("#train_reset_clustering").prop("checked", false);
        }
    });
    // $("#train_embeddings").change(()=>{
    //     if (!$("#train_embeddings").prop("checked")) {
    //         $("#train_reduction").prop("checked", false);
    //         $("#train_clustering").prop("checked", false);
    //         $("#train_summerize").prop("checked", false);
    //     }
    // });
    // $("#train_reduction").change(()=>{
    //     if ($("#train_reduction").prop("checked")) {
    //         $("#train_embeddings").prop("checked", true);
    //     } else {
    //         $("#train_clustering").prop("checked", false);
    //         $("#train_summerize").prop("checked", false);
    //     }
    // });
    // $("#train_clustering").change(()=>{
    //     if ($("#train_clustering").prop("checked")) {
    //         $("#train_embeddings").prop("checked", true);
    //         $("#train_reduction").prop("checked", true);
    //     } else {
    //         $("#train_summerize").prop("checked", false);
    //     }
    // });
    // $("#train_summerize").change(()=>{
    //     if ($("#train_summerize").prop("checked")) {
    //         $("#train_embeddings").prop("checked", true);
    //         $("#train_reduction").prop("checked", true);
    //         $("#train_clustering").prop("checked", true);
    //     }
    // });
    $("#confirm_begin_training").click(() => {
        if (handler) return;
        $.modal.close();
        param = {};
        ["embedding", "reduction", "clustering", "summarize"].forEach((seq) => {
            param["reset_" + seq] = $("#train_reset_" + seq).prop("checked");
            param[seq] = $("#train_" + seq).prop("checked");
        });
        handler = {
            success: (r) => {
                handler = null;
                new TableUpdateHandler().success();
                update_result();
            }
        }
        send_action("train", param);
    });
    $("#confirm_begin_testing").click(() => {
        if (handler) return;
        $.modal.close();
        param = {};
        ["embedding", "reduction", "clustering", "summarize"].forEach((seq) => {
            param["reset_" + seq] = $("#train_reset_" + seq).prop("checked");
            param[seq] = $("#train_" + seq).prop("checked");
        });
        handler = {
            success: (r) => {
                handler = null;
                $("#show_devset").click();
                new TableUpdateHandler().success();
            }
        }
        send_action("test", param);
    });
//#endregion

//#region Dataset manipulation
    $("#show_trainset").click(() => {
        $table = $("#dataset_table").DataTable();
        $table.ajax.url("/ds/train").load();
        $("#show_trainset").addClass("selected");
        $("#show_devset").removeClass("selected");
        viewing_trainset = true;
    });
    $("#show_devset").click(() => {
        $table = $("#dataset_table").DataTable();
        $table.ajax.url("/ds/dev").load();
        $("#show_devset").addClass("selected");
        $("#show_trainset").removeClass("selected");
        viewing_trainset = false;
    });
    $("#copy_selected").click(() => {
        if (handler) return;
        handler = new TableUpdateHandler();
        idx = get_dataset_checked();
        params = {
            rows: idx,
            type: "copy",
            to: viewing_trainset ? "dev" : "train",
        }
        send_action("copy_rows", params);
    });
    $("#move_selected").click(() => {
        if (handler) return;
        handler = new TableUpdateHandler();
        idx = get_dataset_checked();
        params = {
            rows: idx,
            type: "move",
            to: viewing_trainset ? "dev" : "train",
        }
        send_action("copy_rows", params);
    });
    $("#delete_row").click(() => {
        if (handler) return;
        handler = new TableUpdateHandler();
        idx = get_dataset_checked();
        params={"train": idx};
        send_action("delete_row", params);
    });
    $("#clean_dataset").click(() => {
        if (handler) return;
        handler = new TableUpdateHandler();
        params = {
            "set": viewing_trainset ? "train" : "dev"
        }
        send_action("clean_dataset", params);
    });
//#endregion

//#region Predict
    $("#confirm_predict").click(()=>{
        if (handler) return;
        texts = $("#predict_texts").val();
        if (!texts) return;
        $.modal.close();
        handler = {
            "success": (r) => {
                handler = {
                    success: (r) => {
                        handler = null;
                        predict_result = r.prediction;
                        show_prediction_result();
                    }
                };
                send_action("read_predict_result");
            }
        };
        param = {
            'texts': texts.split("\n"),
            'detailed': true,
        }
        send_action("predict", param);
    });
//#endregion

//#region group search
    $("#groups_table_filter").hide();
    $("#group_search").change(() => {
        do_group_search();
    })
//#endregion

//#region download
    $("#download_model").click(() => {
        if (handler) return;
        handler = {
            "success": (r) => {
                handler = null;
                download(r["path"]);
            }
        };
        send_action("save_model");
    });
    $("#download_csv").click(() => {
        if (handler) return;
        handler = {
            "success": (r) => {
                handler = null;
                download(r["path"]);
            }
        };
        send_action("save_to_csv");
    });
//#endregion

//#region Training result view
    $("#table_view").click(() => {
        $("#groups_table_wrapper").show();
        $("#groups_tree").hide();
    })
    $("#tree_view").click(() => {
        $("#groups_table_wrapper").hide();
        $("#groups_tree").show();
    })
//#endregion

//#region Visualize Button
    $("#visualize_group_counts").click(() => {
        visualize_group_count();
    });
    $("#visualize_scatters").click(() => {
        if (handler != null) return;
        handler = {
            success: (r) => {
                handler = null;
                vis_data = {
                    "train": r.train,
                    "dev": r.dev,
                }
                //loading_dialog("Rendering");
                create_scatter_table();
                //hide_loading_dialog();
            }
        }
        send_action("text_summary");
    });
    $("#visualize_timeline").click(() => {
        if (handler != null) return;
        handler = {
            success: (r) => {
                handler = null;
                vis_data = {
                    "counts": r.counts,
                }
                create_timeline();
            }
        }
        send_action("group_over_time");
    })
//#endregion

    $("#log_clear").click((e) => {
        $("#logs").html('<input type="button" name="log_clear" id="log_clear" value="Clear Log" />');
    });

//#region checkpoints
    $("#save_checkpoint").click(()=>{
        if (handler) return;
        handler = {};
        send_action("save_checkpoint");
    });
    $("#load_checkpoint").click(()=>{
        if (handler) return;
        $dlg = $("#load_checkpoint_dialog section").html("");
        handler = {
            success: (r) => {
                handler = null;
                if (r.checkpoints.length) {
                    r.checkpoints.forEach((v, i) => {
                        $span = $('<span>').appendTo($dlg);
                        $('<a class="chkpoint" href="#"></a>').html(v).appendTo($span);
                        $('<a class="del_chkpt" href="#">del</a>').data("chkpoint", v).appendTo($span);
                    });
                    $(".chkpoint").click((e)=>{
                        if (handler) return;
                        $.modal.close();
                        handler = {
                            success: () => {
                                handler = null;
                                new TableUpdateHandler().success();
                                setTimeout(()=> {
                                    update_result();
                                }, 100);
                            }
                        }
                        param = {
                            "name": $(e.delegateTarget).html(),
                        }
                        send_action("load_checkpoint", param);
                    });
                    $(".del_chkpt").click((e)=>{
                        if (handler) return;
                        $.modal.close();
                        handler = {}
                        param = {
                            "name": $(e.delegateTarget).data("chkpoint"),
                        }
                        send_action("delete_checkpoint", param);
                    });
                } else {
                    $("#load_checkpoint_dialog").html('<span class="no_chkpoint">No checkpoints</span>');
                }
                $("#hb_checkpoint").click();
            }
        };
        send_action("list_checkpoint");
    });
    $("#new_model").click(()=>{
        if (handler) return;
        handler = {
            success: () => {
                summary_table = null;
                summary_tree = null;
                predict_result = null;
                vis_data = null;
                group_filter = null;
            }
        }
        send_action("new_model");
    })
//#endregion

    // lazy load model summary if any
    setTimeout(()=> {
        update_result();
    }, 1000);
}

$(document).ready(() => {
    init();
});

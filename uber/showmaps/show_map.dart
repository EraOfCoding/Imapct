//import 'package:google_maps/';
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:sliding_up_panel/sliding_up_panel.dart';


class ShowSimpleMap extends StatefulWidget {
  const ShowSimpleMap({Key? key}) : super(key: key);

  @override
  State<ShowSimpleMap> createState() => _ShowSimpleMapState();
}

class _ShowSimpleMapState extends State<ShowSimpleMap> {
  late GoogleMapController mapController;
  static final LatLng _center = const LatLng( 34.020882,  -6.841650);
  void _onMapCreated(GoogleMapController controller){
    mapController = controller;
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // appBar: AppBar(
      //   title: Text("winning team app"),),
      body:Stack(
        children:<Widget>[  Container(
          height: MediaQuery.of(context).size.height,
          width: MediaQuery.of(context).size.width,
          child: GoogleMap(
              markers: {testMarker},
              mapType: MapType.normal ,
              onMapCreated:_onMapCreated,
              initialCameraPosition: CameraPosition(target: _center, zoom: 11.1),
            ),
        ),
          slidingPage()

        ]
    )
    );

  }
  Marker testMarker = Marker(markerId: MarkerId("test test"),
      position: _center,
      infoWindow: InfoWindow(title: "Destination", snippet: "this is your destination" ),
      icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueAzure));

  TextEditingController textController = TextEditingController();
  Widget userInput(){
    return Row(
      children: <Widget>[Icon(Icons.search),
        Container(
          height: 50,
          width: 200,
          child: TextField(
            controller: textController,
            decoration: InputDecoration(
              hintText: "enter your destination",
            ),

          ),
        ),
      ],
    );
  }

  Widget slidingPage(){
  return SlidingUpPanel(
    panel:
     Column(
       children: <Widget>[
         Padding(
           padding: const EdgeInsets.all(8.0),
           child: Column(
             mainAxisAlignment: MainAxisAlignment.spaceAround,
             children: <Widget>[
               Icon(Icons.energy_savings_leaf, color: Colors.green),
               dropdownbutton("Car"),
               dropdownbutton("Model"),
               dropdownbutton("Class")

             ],
           ),
         ),
       ],
     ),
    borderRadius: BorderRadius.circular(15),
    backdropEnabled: true,


  );
  }
  String selectedCar = "Car";
  Widget dropdownbutton(String choice,){
    return Column(
      children: <Widget>[
         Center(child: Text(choice))
        ,
        DropdownButton<String>(items: <String>["ASTON MARTIN", "AUDI", "Car"].map<DropdownMenuItem<String>>((String value) {
          return DropdownMenuItem<String>(
              value: value,
              child:Text(value));
        }).toList(),
          onChanged: (String? newValue){
          setState(() {
            selectedCar = newValue!;
          });
        }, value: selectedCar,hint: Text("Car"),),

      ],
    );
  }
}


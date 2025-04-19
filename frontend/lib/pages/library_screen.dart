
import 'package:flutter/material.dart';

class LibraryScreen extends StatefulWidget {
  const LibraryScreen({super.key});

  @override
  State<LibraryScreen> createState() => _LibraryScreenState();
}

class _LibraryScreenState extends State<LibraryScreen> {
  final TextEditingController searchController = TextEditingController();
  final List<MusicItem> musicItems = List.generate(
    20, 
    (index) => MusicItem(
      title: 'Music Name',
      type: 'Music Type',
      isFavorite: false,
    ),
  );

  @override
  Widget build(BuildContext context) {
    // Removed unused 'size' variable
    
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Color(0xFFD81B60),
              Color(0xFF880E4F),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // App Bar with Home text and profile
              _buildAppBar(),
              
              // Search bar
              _buildSearchBar(),
              
              // Your Library header
              _buildLibraryHeader(),
              
              // All filter button
              _buildFilterButton(),
              
              // Music list
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: ListView.builder(
                    itemCount: musicItems.length,
                    itemBuilder: (context, index) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 10.0),
                        child: _buildMusicCard(musicItems[index], index),
                      );
                    },
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Logo on left
          Container(
            height: 50,
            width: 50,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(4),
            ),
            padding: EdgeInsets.all(4),
            child: Image.asset('assets/images/Frame 1.png'),
          ),
          
          // HOME text
          Text(
            'HOME',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
          
          // Profile avatar
          CircleAvatar(
            backgroundColor: Colors.white,
            radius: 16,
            child: Icon(Icons.person, color: Colors.pink[900], size: 20),
          ),
        ],
      ),
    );
  }

  Widget _buildSearchBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(4),
        ),
        padding: EdgeInsets.symmetric(horizontal: 8),
        child: TextField(
          controller: searchController,
          decoration: InputDecoration(
            hintText: 'Search here...',
            border: InputBorder.none,
            icon: Icon(Icons.search, color: Colors.grey),
          ),
        ),
      ),
    );
  }

  Widget _buildLibraryHeader() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Container(
        width: double.infinity,
        padding: EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(4),
        ),
        child: Center(
          child: Text(
            'YOUR LIBRARY',
            style: TextStyle(
              color: Colors.pink[900],
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildFilterButton() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Align(
        alignment: Alignment.centerLeft,
        child: Container(
          padding: EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          decoration: BoxDecoration(
            color: Colors.black,
            borderRadius: BorderRadius.circular(4),
          ),
          child: Text(
            'All',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMusicCard(MusicItem item, int index) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          // Left thumbnail - landscape image
          ClipRRect(
            borderRadius: BorderRadius.only(
              topLeft: Radius.circular(10),
              bottomLeft: Radius.circular(10),
            ),
            child: Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.lightBlue[200]!, Colors.green[200]!],
                ),
              ),
              child: Stack(
                children: [
                  Positioned(
                    top: 10,
                    left: 10,
                    child: Icon(Icons.cloud, color: Colors.white, size: 18),
                  ),
                  Positioned(
                    top: 5,
                    right: 10,
                    child: Icon(Icons.cloud, color: Colors.white, size: 14),
                  ),
                ],
              ),
            ),
          ),
          
          // Music details
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.title,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                  Text(
                    item.type,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                    ),
                  ),
                  Divider(height: 8),
                  Text(
                    'Duration',
                    style: TextStyle(
                      fontSize: 10,
                      color: Colors.grey,
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          // Favorite button
          IconButton(
            icon: Icon(
              item.isFavorite ? Icons.favorite : Icons.favorite_border,
              color: Colors.pink[300],
            ),
            onPressed: () {
              setState(() {
                musicItems[index] = MusicItem(
                  title: item.title,
                  type: item.type,
                  isFavorite: !item.isFavorite,
                );
              });
            },
          ),
        ],
      ),
    );
  }
}

class MusicItem {
  final String title;
  final String type;
  final bool isFavorite;

  MusicItem({
    required this.title,
    required this.type,
    required this.isFavorite,
  });
}
